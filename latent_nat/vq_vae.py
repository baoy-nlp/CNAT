import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules.transformer_sentence_encoder import init_bert_params, LayerDropModuleList

from nat_base.vanilla_nat import ensemble_decoder, NAT, NATDecoder
from .global_names import *
from .model_utils import compute_margin
from .posterior import CODE_CLS
from .predictor import NARPredictor


class GateNet(nn.Module):
    def __init__(self, d_model, d_hidden, d_output, dropout=0.0):
        super().__init__()
        self.input_to_hidden = nn.Linear(d_model, d_hidden)
        self.hidden_to_output = nn.Linear(d_hidden, d_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        h = F.relu(self.input_to_hidden(inputs))
        h = self.dropout(h)
        return self.hidden_to_output(h)


@register_model("vq_vae")
class VectorQuantizedVAE(NAT):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        NAT.add_args(parser)
        NARPredictor.add_args(parser)
        Decoder.add_args(parser)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = Decoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
            extra_ret=True,
            use_oracle=True,
        )

        extra = None
        if tgt_tokens is not None:
            word_ins_out, extra = word_ins_out[0], word_ins_out[1]

        model_ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }
        }

        if extra is not None:
            model_ret = self.update_quantize_vector(extra, model_ret)

        return model_ret

    def update_quantize_vector(self, extra, model_ret):
        latent_ret = extra[LATENT_RET]
        model_ret = self.decoder.compute_loss(latent_ret, model_ret)
        if self.training:
            if not (getattr(self.args, "finetune", False) and getattr(self.args, "no_use_ema", False)):
                self.decoder.update_codes(latent_ret)
        return model_ret

    def step_strategy(self, step=0):
        self.decoder.update_gradually_ratio(step)

    def epoch_strategy(self):
        if hasattr(self.decoder.codes, "reinit") and getattr(self.args, "epoch_reinit", False):
            self.decoder.codes.reinit()

        if getattr(self.args, "ema_switch", False):
            setattr(self.args, "no_use_ema", not getattr(self.args, "no_use_ema", False))
            print("epoch-switch: from {} to {}".format(
                not getattr(self.args, "no_use_ema"),
                getattr(self.args, "no_use_ema"))
            )


class Decoder(NATDecoder):
    """
    - Posterior q(z|x,y) --- embedding(y) + vq_codes
    - Prior p(z|x) --- vq_predictor
    - Reconstructor p(y|x,z)
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.ret_score, self.codes = self.build_vector_codes(args, self.embed_tokens)

        self.prior = NARPredictor(
            args,
            padding_idx=self.padding_idx,
            embed_positions=self.embed_positions,
            embed_tokens=None if getattr(args, "no_share_code", False) else self.codes.embedding,
        )
        self.set_pad_zero = getattr(args, "set_pad_zero", False)

        # used for training
        self.alpha = args.vq_alpha
        self.beta = args.vq_beta
        self.vq_kl = getattr(args, "vq_kl", 0.0)
        self.sg_vq_pred = args.sg_vq_pred

        self.only_ema = getattr(args, "only_ema", False)
        self.use_info_bound = getattr(args, "use_info_bound", False)
        self.info_z = getattr(args, "info_z", 0.)
        self.info_bound_z = math.log(self.codes.embedding.num_embeddings) if self.use_info_bound else -1

        self.use_info_exp = getattr(args, "use_info_exp", False)

        if getattr(args, "backward_enc", 0) > 0:
            # for q(z|y,x)
            enc_num_layer = getattr(args, "backward_enc", 0)
            if self.decoder_layerdrop > 0.0:
                self.backward_encoder = LayerDropModuleList(p=self.decoder_layerdrop)
            else:
                self.backward_encoder = nn.ModuleList([])
            for i in range(enc_num_layer):
                self.backward_encoder.append(self.build_decoder_layer(args, no_encoder_attn))
            share_num_layer = getattr(args, "share_forward_backward_enc")
            if share_num_layer > 0:
                for i in range(share_num_layer):
                    self.backward_encoder[i] = self.layers[i]
        else:
            self.backward_encoder = None

        self.gradually = getattr(args, "gradually", False)
        self.gradually_ratio = getattr(args, "gradually_ratio", 0.0)

        self.gated_func = getattr(args, "gated_func", "residual")
        if self.gated_func.startswith("gated"):
            self.gate = GateNet(
                d_model=self.embed_dim * 2, d_hidden=self.embed_dim * 4, d_output=self.embed_dim,
                dropout=args.dropout
            )
        else:
            self.gate = None

        self.layer_gated_func = getattr(args, "layer_gated_func", "None")
        if self.layer_gated_func == "gated":
            self.layer_gate = GateNet(
                d_model=self.embed_dim * 2,
                d_hidden=self.embed_dim * 4,
                d_output=self.embed_dim,
                dropout=args.dropout
            )
        else:
            self.layer_gate = None

        self.remove_pos = getattr(args, "remove_pos", False)
        self.gradually_schedule = getattr(args, "gradually_schedule", "pred")

        self.adaptive_steps = getattr(args, "adaptive_steps", -1)
        self.adaptive_start = getattr(args, "adaptive_start", 1.0)
        self.adaptive_end = getattr(args, "adaptive_end", 1.0)

        self.hybrid_scale = getattr(args, "hybrid_scale", 0.0)

    @classmethod
    def build_vector_codes(cls, args, embed_token=None):
        code_cls = getattr(args, "code_cls", "code")
        code = CODE_CLS[code_cls](
            num_codes=args.vq_num,
            code_dim=args.decoder_embed_dim,
            lamda=args.vq_lamda,
            not_update=getattr(args, "only_ema", False),
            embed_token=embed_token if getattr(args, "vq_init", False) else None
        )
        if code_cls == "code":
            return False, code
        else:
            return True, code

    @staticmethod
    def add_args(parser):
        parser.add_argument("--vq-alpha", type=float, help="weights on the vq predictor loss")
        parser.add_argument("--vq-beta", type=float, help="weights on the vector quantized loss")

        parser.add_argument("--vq-kl", type=float, help="weights on the vector quantized loss")
        parser.add_argument("--vq-num", type=int, help="hyper-parameter, number of categorical")
        parser.add_argument("--vq-lamda", type=float, help="weight of exponential moving average loss")
        parser.add_argument("--sg-vq-pred", action="store_true", default=False,
                            help="stop the gradients back-propagated from the vq predictor")
        parser.add_argument("--vq-init", action="store_true", help="whether init the vq code books")
        parser.add_argument("--vq-learn-code", action="store_true",
                            help="use the dot product for nearest search")

        parser.add_argument("--code-cls", type=str, default="code", choices=CODE_CLS.keys(),
                            help="which code cls is used for vector quantize")
        parser.add_argument("--no-share-code", action="store_true",
                            help="do not share the embedding between the code and prior")
        parser.add_argument("--no-use-ema", action="store_true",
                            help="remove the exponential moving average for latent codes")
        parser.add_argument("--only-ema", action="store_true",
                            help="update the code with only exponential moving average")

        parser.add_argument("--set-pad-zero", action="store_true", help="set decoder pad code to zeros")

        parser.add_argument("--use-info-bound", action="store_true")
        parser.add_argument("--use-info-exp", action="store_true")
        parser.add_argument("--info-z", type=float, default=0.0)

        parser.add_argument("--backward-enc", type=int, default=0, help="posterior encoder")
        parser.add_argument("--share-forward-backward-enc", type=int, default=0,
                            help="share the forward decoding and backward encoding")

        parser.add_argument("--gradually", action="store_true")
        parser.add_argument("--gradually-ratio", type=float, default=0.5)
        parser.add_argument("--gradually-schedule", type=str, default="pred", choices=["pred", "glancing", "random"])
        parser.add_argument("--adaptive-start", type=float, default=0.5)
        parser.add_argument("--adaptive-end", type=float, default=0.3)
        parser.add_argument("--adaptive-steps", type=int, default=-1)

        parser.add_argument("--gated-func", type=str, default="residual")
        parser.add_argument("--layer-gated-func", type=str, default="none")
        parser.add_argument("--remove-pos", action="store_true")

        parser.add_argument("--st-scale", type=float, default=0.0,
                            help="straight through scale for mapping X to Y")
        parser.add_argument("--vst-scale", type=float, default=0.0,
                            help="straight through scale for mapping Z to Y")
        parser.add_argument("--hybrid-scale", type=float, default=0.0)

        parser.add_argument("--epoch-reinit", action="store_true", default=False)
        parser.add_argument("--epoch-every", type=int, default=1)
        parser.add_argument("--ema-switch", action="store_true", default=False)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, tgt_tokens=None, extra_ret=False,
                use_oracle=True, **unused):
        features, ret = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            tgt_tokens=tgt_tokens,
            use_oracle=True,
            **unused
        )
        decoder_out = self.output_layer(features)
        decoder_out = F.log_softmax(decoder_out, -1) if normalize else decoder_out
        # if tgt_tokens is not None and extra_ret:
        if extra_ret:
            return decoder_out, ret
        else:
            return decoder_out

    def extract_features(self, prev_output_tokens, encoder_out=None, early_exit=None, embedding_copy=False,
                         tgt_tokens=None, use_oracle=True, **unused):

        x, decoder_padding_mask, pos = self.forward_decoder_inputs(prev_output_tokens, encoder_out=encoder_out)

        # x including position information

        def _gate_connect(_x, _latent):
            if self.gate is not None:
                g = self.gate(torch.cat([_x, _latent], dim=-1))
                if self.gated_func.endswith("tanh"):
                    g = g.tanh()
                    _x = _x * g + _latent * (1 - g)
                elif self.gated_func.endswith("st"):
                    _x = g.tanh()
                else:
                    g = g.sigmoid()
                    _x = _x * g + _latent * (1 - g)
            else:
                _x = _x + _latent
            return _x

        def hybrid_oracle(_x, _c, _mask):
            _mask = _mask.float().unsqueeze(-1)
            return _c * _mask + _x * (1 - _mask)

        residual = x
        z, code_ret = self.forward_codes(
            inputs=x,
            decoder_padding_mask=decoder_padding_mask,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
            use_oracle=use_oracle
        )
        x = z if self.remove_pos else z + pos
        x = _gate_connect(residual, x)

        if self.hybrid_scale > 0. and code_ret["oracle"] is not None and self.training:
            oracle_mask = code_ret["oracle"]
            oracle_x = hybrid_oracle(_x=residual, _c=x, _mask=oracle_mask)
            oracle_x = oracle_x.transpose(0, 1)
            oracle_features, _, _ = self._reconstruct(oracle_x, decoder_padding_mask, encoder_out, early_exit)
            oracle_out = self.output_layer(oracle_features)
            code_ret["oracle_out"] = oracle_out

        x = x.transpose(0, 1)  # prepare for attentive decoding
        x, attn, inner_states = self._reconstruct(x, decoder_padding_mask, encoder_out, early_exit)

        return x, {"attn": attn, "inner_states": inner_states, LATENT_RET: code_ret}

    def _reconstruct(self, x, decoder_padding_mask, encoder_out, early_exit=None):
        attn = None
        inner_states = [x]
        for i, layer in enumerate(self.layers):  # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)
        if self.layer_norm:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)  # T x B x C -> B x T x C
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        return x, attn, inner_states

    def forward_codes(self, inputs, decoder_padding_mask, encoder_out, tgt_tokens, use_oracle=True):
        posterior_ret = self._posterior(inputs, decoder_padding_mask, encoder_out, tgt_tokens)
        prior_out, prior_ret = self._prior(inputs, decoder_padding_mask, encoder_out, tgt_tokens=posterior_ret['tgt'])

        oracle = None
        if tgt_tokens is not None and (use_oracle or not self.training):
            z = posterior_ret["z_q_st"]  # batch_size, seq_len, depth
            if self.gradually and self.training:
                z, oracle = self.schedule_sampling(prior_out, posterior_ret['tgt'], z, decoder_padding_mask)
        else:
            z = self.codes.forward_embedding(prior_out.token)  # batch_size, seq_len, depth

        if self.only_ema:
            z = z.detach()

        if self.set_pad_zero and decoder_padding_mask.sum().item() > 0:
            z = (1 - decoder_padding_mask.float()).unsqueeze(-1) * z

        return z, {"prior": prior_ret, "posterior": posterior_ret, "prior_out": prior_out, "oracle": oracle}

    def schedule_sampling(self, prior_out, tgt, tgt_embed, padding_mask=None, use_gumbel=False):
        def _predicted_sampling():
            # sampling with the prob
            target_score = tgt.clone().float().uniform_()
            sample = target_score.lt(prob)  # sampling predicted
            mask = sample.float().unsqueeze(-1)
            return mask * predict_embed + (1 - mask) * tgt_embed, ~sample

        def _random_sampling():
            target_score = tgt.clone().float().uniform_()  # sampling
            target_score.masked_fill_(padding_mask, 2.0)  # set 2.0 for padding token
            sample_length = (~padding_mask.long()).sum(1).float() * self.gradually_ratio + 1

            _, target_rank = target_score.sort(1)
            target_cutoff = utils.new_arange(target_rank) < sample_length[:, None].long()

            sample = target_cutoff.scatter(1, target_rank, target_cutoff)  # sampling reference
            mask = sample.float().unsqueeze(-1)

            return (1 - mask) * predict_embed + mask * tgt_embed, sample

        def _glancing_sampling():
            ill_match = predict.ne(tgt) * (~padding_mask).long()  # compute the distance of pred and tgt
            target_score = tgt.clone().float().uniform_()  # sampling
            target_score.masked_fill_(padding_mask, 2.0)  # set 2.0 for padding token
            sample_length = ill_match.sum(1).float() * self.gradually_ratio + 1  # determine the target length
            _, target_rank = target_score.sort(1)
            target_cutoff = utils.new_arange(target_rank) < sample_length[:, None].long()

            sample = target_cutoff.scatter(1, target_rank, target_cutoff)  # sampling reference
            mask = sample.float().unsqueeze(-1)
            return (1 - mask) * predict_embed + mask * tgt_embed, sample

        score = prior_out.out.softmax(dim=-1)  # batch_size, seq_len, vocab
        prob, predict = score.max(dim=-1)

        predict_embed = self.codes.forward_embedding(predict)  # batch_size, seq_len, depth

        if self.gradually_schedule == "pred":
            embed, ref = _predicted_sampling()
        elif self.gradually_schedule == "glancing":
            embed, ref = _glancing_sampling()
        elif self.gradually_schedule == "random":
            embed, ref = _random_sampling()
        else:
            raise RuntimeError("gradually schedule is wrong!")

        match = predict.eq(tgt)

        oracle = (match.long() + ref.long()).bool()

        return embed, oracle

    def update_gradually_ratio(self, step=0):
        if step <= self.adaptive_steps:
            delta = (self.adaptive_start - self.adaptive_end) / self.adaptive_steps
            self.gradually_ratio = self.adaptive_start - step * delta

    def update_codes(self, ret):
        posterior = ret["posterior"]
        z_enc = posterior['z_e'].view(-1, posterior['z_e'].size(-1))  # batch_size, sequence_length, D
        enc_sum = self.codes.forward_count(z_enc, posterior["mask"], posterior["tgt"])
        self.codes.exponential_moving_average(enc_sum)

    def compute_loss(self, latent_ret, model_ret):
        self._code_term(latent_ret, model_ret)
        self._kl_term(latent_ret, model_ret)  # KL( p||q )

        if self.alpha > 0.:
            prior_ret = latent_ret['prior']
            for key, loss in prior_ret.items():
                if isinstance(loss, dict):
                    if "factor" in loss:
                        loss["factor"] = loss["factor"] * self.alpha
                    else:
                        loss["factor"] = self.alpha
                    if "loss" in loss:
                        loss["loss"] = loss["loss"] * self.alpha
                    model_ret[key] = loss

        if self.hybrid_scale > 0. and "oracle_out" in latent_ret:
            model_ret['oracle'] = {
                "out": latent_ret["oracle_out"],
                "tgt": model_ret["word_ins"]["tgt"],
                "mask": latent_ret['oracle'],
                "factor": self.hybrid_scale
            }

        if not hasattr(self, 'reg_func') and self.info_z > 0:
            self._informative_z(latent_ret, model_ret)

        return model_ret

    def _prior(self, inputs, decoder_padding_mask, encoder_out, tgt_tokens=None):
        if self.sg_vq_pred:
            inputs = inputs.detach()
            encoder_out = EncoderOut(
                encoder_out=encoder_out.encoder_out.detach() if encoder_out.encoder_out is not None else None,
                encoder_padding_mask=encoder_out.encoder_padding_mask,
                encoder_states=encoder_out.encoder_states,
                encoder_embedding=encoder_out.encoder_embedding,
                src_tokens=encoder_out.src_tokens,
                src_lengths=encoder_out.src_lengths
            )

        return self.prior(
            inputs=inputs,
            mask=decoder_padding_mask,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
            log_prob=(self.info_z > 0)
        )

    def _posterior(self, inputs, decoder_padding_mask, encoder_out, tgt_tokens):
        # from y to latent
        z_q_st, z_q, z_e, tgt, score = None, None, None, None, None

        if tgt_tokens is not None:
            if self.backward_encoder is None:
                z_e = self.embed_tokens(tgt_tokens)
            else:
                z_e = self._posterior_encoding(
                    prev_output_tokens=tgt_tokens,
                    encoder_out=encoder_out
                )
            if self.ret_score:
                z_q_st, z_q, tgt, score = self.codes.straight_through(z_e)
            else:
                z_q_st, z_q, tgt = self.codes.straight_through(z_e)
                score = None

        return {
            "z_q_st": z_q_st,
            "z_q": z_q,
            "z_e": z_e,
            "tgt": tgt,
            "out": score,
            "mask": ~decoder_padding_mask,
        }

    def _posterior_encoding(self, prev_output_tokens, encoder_out):
        x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)
        x = x.transpose(0, 1)

        for i, layer in enumerate(self.backward_encoder):
            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1).contiguous()
        return x

    def _code_term(self, latent_ret, model_ret):
        # reconstruct for embedding
        posterior = latent_ret["posterior"]
        if self.beta > 0.:
            z_enc = posterior['z_e'].view(-1, posterior['z_e'].size(-1))  # batch_size, sequence_length, D
            z_code = posterior['z_q'].view(-1, z_enc.size(-1))  # batch_size, sequence_length, D
            mask = posterior["mask"].float()
            count = mask.sum()
            code2enc = (F.mse_loss(z_code, z_enc.detach(), reduction='none').sum(dim=-1).contiguous().view(
                *mask.size()) * mask).sum() / count * self.beta

            enc2code = (F.mse_loss(z_enc, z_code.detach(), reduction='none').sum(dim=-1).contiguous().view(
                *mask.size()) * mask).sum() / count * self.beta
            loss_ret = {
                "code2enc": {"loss": code2enc, "factor": self.beta},
                "enc2code": {"loss": enc2code, "factor": self.beta}
            }
            model_ret.update(loss_ret)
        return model_ret

    def _kl_term(self, latent_ret, model_ret):
        if self.vq_kl > 0:
            posterior = latent_ret["posterior"]
            prior = latent_ret["prior"]
            mask = posterior["mask"]
            q = posterior['out']
            if q is not None:
                if "log_prob" in prior:
                    p = prior["log_prob"][mask].exp()  # used for CRF
                else:
                    p = prior["out"][mask].softmax(dim=-1)
                log_q = q[mask].log_softmax(dim=-1)  # batch_size, seq_len, vocab
                model_ret["vq-KL"] = {
                    "out": log_q,
                    "loss": F.kl_div(log_q, p, reduction="mean") * self.vq_kl,
                    "factor": self.vq_kl,
                    "no-acc": True
                }
        return model_ret

    def _informative_z(self, latent_ret, model_ret):
        if not getattr(self, "info_z", 0) > 0:
            return model_ret
        posterior = latent_ret["posterior"]
        prior = latent_ret["prior"]

        qzy = None if "out" not in posterior else posterior["out"]
        mask = posterior["mask"]
        if "log_prob" in prior:  # mean used CRF
            pzx = prior["log_prob"].exp()
            log_diff = False
        else:
            pzx = prior["out"]
            log_diff = True
        loss = compute_margin(
            dist1=qzy,
            dist2=pzx,
            mask=mask,
            tgt=posterior["tgt"],
            log_diff=log_diff,
            use_info_exp=self.use_info_exp and log_diff,
            info_bound=self.info_bound_z if log_diff else -1,
            is_prob="log_prob" in prior
        )
        model_ret["info-z"] = {
            "loss": loss.mean() * self.info_z,
            "factor": self.info_z,
        }

        return model_ret


def base_architecture(args):
    from nat_base.vanilla_nat import base_architecture
    base_architecture(args)

    args.extra_embed_dim = getattr(args, "extra_embed_dim", args.decoder_embed_dim)

    # parameter for vector quantized
    args.predictor_layers = getattr(args, "predictor_layers", 3)
    args.predictor_learned_pos = getattr(args, "predictor_learned_pos", args.decoder_learned_pos)
    args.share_learned_pos = getattr(args, "share_learned_pos", True)

    args.vq_alpha = getattr(args, "vq_alpha", 0.)
    args.vq_beta = getattr(args, "vq_beta", 0.)
    args.vq_num = getattr(args, "vq_num", 50)
    args.vq_lamda = getattr(args, "vq_lamda", 0.999)


@register_model_architecture("vq_vae", "vq_vae_wmt14")
def wmt14_en_de(args):
    base_architecture(args)


@register_model_architecture('vq_vae', 'vq_vae_iwslt16')
def iwslt16_de_en(args):
    from nat_base.vanilla_nat import nat_iwslt16_de_en
    nat_iwslt16_de_en(args)
    base_architecture(args)


@register_model_architecture('vq_vae', 'vq_vae_iwslt14')
def iwslt14_de_en(args):
    from nat_base.vanilla_nat import nat_iwslt14_de_en
    nat_iwslt14_de_en(args)
    base_architecture(args)


@register_model_architecture('vq_vae', 'vq_vae_base')
def nat_base(args):
    from nat_base.vanilla_nat import nat_base
    nat_base(args)
    base_architecture(args)
