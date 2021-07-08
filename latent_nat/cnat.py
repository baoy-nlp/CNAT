import copy

import torch
import torch.nn as nn
from fairseq.models import register_model, register_model_architecture

from latent_nat.glat import ReferenceSampler
from latent_nat.predictor import CRFPredictor
from latent_nat.utils import GlobalNames
from latent_nat.vector_quantization import vq_st, vq_search
from latent_nat.vnat import VNATDecoder, VariationalNAT, init_bert_params

INF: float = 1e-10


@register_model("cnat")
class QuantizeNAT(VariationalNAT):
    """
    Incorporate Discrete Latent Variable for NAT
    """

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = QuantizeNATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @staticmethod
    def add_args(parser):
        VariationalNAT.add_args(parser)
        QuantizeNATDecoder.add_args(parser)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        word_ins_out, inner_states = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
            extra_ret=True,
        )
        losses = self._compute_loss(word_ins_out, tgt_tokens, encoder_out, inner_states)

        if inner_states is not None:
            losses.update(self._compute_vq_loss(inner_states))

            # TODO: only update in the local mini-batch, can not synchronized update in multi-gpu or multi update-freq
            if self.training and getattr(self.args, "vq_ema", False):
                self.decoder.update_code(inner_states[GlobalNames.POST_RET])

        return losses

    def _compute_vq_loss(self, inner_states):
        latent_factor = getattr(self.args, "latent_factor", 1.0)
        _losses = {}
        if "VQ" in inner_states[GlobalNames.PRI_RET]:
            vq = inner_states[GlobalNames.PRI_RET]["VQ"]
            if "factor" in vq:
                vq["factor"] = vq.get("factor", 1.0) * latent_factor
            if "loss" in vq:
                vq["loss"] = vq["loss"] * vq["factor"]
            _losses["VQ"] = vq

        if "VQ-Out" in inner_states[GlobalNames.PRI_RET]:
            vq_out = inner_states[GlobalNames.PRI_RET]["VQ-Out"]
            if "factor" in vq_out:
                vq_out["factor"] = vq_out["factor"] * latent_factor
            if "loss" in vq_out:
                vq_out["loss"] = vq_out["loss"] * vq_out["factor"]
            _losses["VQ-Out"] = vq_out
        return _losses


class QuantizeNATDecoder(VNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.vq_schedule_ratio = args.vq_schedule_ratio
        self.is_schedule_z = self.vq_schedule_ratio > 0.
        self.z_sampler = ReferenceSampler(num_mode="adaptive", sub_mode="schedule")

    @staticmethod
    def add_args(parser, key=None):
        # for vector quantization
        parser.add_argument("--num-codes", type=int)
        parser.add_argument("--lamda", type=float, default=0.999,
                            help="use for exponential moving average")
        parser.add_argument("--vq-ema", action="store_true")
        parser.add_argument("--vq-dropout", type=float)
        parser.add_argument("--vq-share-input-output-embed", action="store_true")
        parser.add_argument("--vq-schedule-ratio", type=float, default=0.0)

        # parameter for predictor
        parser.add_argument(
            "--share-bottom-layers", action="store_true", help="share bottom layer of NAT decoder and latent decoder"
        )
        parser.add_argument("--vq-self-attn-cls", type=str)
        parser.add_argument("--vq-block-cls", type=str)
        parser.add_argument("--crf-cls", type=str)
        parser.add_argument("--crf-num-head", type=int)

    def update_code(self, posterior):
        self.posterior.update_code(posterior)

    def forward_latent(self, encoder_out, tgt_tokens=None, inputs=None, decoder_padding_mask=None, **unused):
        if tgt_tokens is not None:
            # vector quantization from the reference --- non-parameter posterior
            inference_out, idx = self._quantize_z(inputs, decoder_padding_mask, tgt_tokens)
        else:
            inference_out, idx = None, None

        if self.prior is None:
            # non-parameterize predictor, nearest search with decoder inputs
            predict_out, idx = self._quantize_z(inputs, decoder_padding_mask)
        else:
            # parameterize predictor, we use NAT-CRF here.
            predict_out, idx = self._predict_z(inputs, decoder_padding_mask, encoder_out, tgt=idx, out=inference_out)

        if inference_out is not None:
            q = predict_out["z_inputs"]
        else:
            q = self.posterior(indices=idx)

        return q, {GlobalNames.PRI_RET: predict_out, GlobalNames.POST_RET: inference_out}

    def _quantize_z(self, inputs, decoder_padding_mask, tgt_tokens=None):
        """ vector quantization for Z"""
        if tgt_tokens is not None:
            inputs = self.forward_embedding(tgt_tokens, add_position=False)[0]

        z_q_st, z_q, idx = self.posterior.straight_through(inputs)

        return {
                   "code_st": z_q_st,
                   "code": z_q,
                   "input": inputs,
                   "tgt": idx,
                   "mask": ~decoder_padding_mask,
               }, idx

    def _predict_z(self, inputs, decoder_padding_mask, encoder_out, tgt=None, out=None):
        """ predict the latent variables """
        outputs, ret = self.prior(
            inputs=inputs,
            decoder_padding_mask=decoder_padding_mask,
            encoder_out=encoder_out,
            tgt_tokens=tgt,
            include_pos=self.args.self_attn_cls != "shaw"
        )
        pred_embed = self.posterior(indices=outputs.token)
        z_inputs = pred_embed
        if out is not None:
            if self.is_schedule_z:
                ref_embed = out["code_st"]
                sample = self.z_sampler.forward_sampling(
                    targets=tgt,
                    padding_mask=decoder_padding_mask,
                    ratio=self.vq_schedule_ratio,
                    logits=ret["out"],
                    s_mode="mixing"
                )
                observed = sample.float().unsqueeze(-1)
                z_inputs = self.z_sampler.forward_inputs(
                    inputs=inputs,
                    ref=ref_embed,
                    observed=observed,
                    pred=pred_embed,
                    s_mode="schedule"
                )
            else:
                z_inputs = out["code_st"]

        ret["z_inputs"] = z_inputs
        return ret, outputs.token

    def _build_posterior(self):
        model_args = self.args
        args = copy.deepcopy(model_args)
        code: EMACode = EMACode(num_codes=args.num_codes, code_dim=args.decoder_embed_dim, lamda=args.lamda)
        return code

    def _build_prior(self):
        main_args = self.args
        args = copy.deepcopy(main_args)
        args.share_decoder_input_output_embed = getattr(main_args, "vq_share_input_output_embed",
                                                        self.share_input_output_embed)
        args.decoder_layers = getattr(main_args, "latent_layers", main_args.decoder_layers)
        args.dropout = getattr(main_args, "vq_dropout", main_args.dropout)
        args.self_attn_cls = getattr(main_args, "vq_self_attn_cls", main_args.self_attn_cls)
        args.block_cls = getattr(main_args, "vq_block_cls", main_args.block_cls)
        # args.decoder_learned_pos = True if args.self_attn_cls != "shaw" else False

        latent_decoder = CRFPredictor(
            args,
            dictionary=Dictionary(num_codes=args.num_codes),
            embed_tokens=self.posterior.embedding if args.share_decoder_input_output_embed else nn.Embedding(
                num_embeddings=args.num_codes, embedding_dim=args.decoder_embed_dim, padding_idx=-1
            ),
            no_encoder_attn=False
        )

        if getattr(args, "share_bottom_layers", False):
            shared_layers = args.latent_layers if args.decoder_layers > args.latent_layers else args.decoder_layers
            for i in range(shared_layers):
                latent_decoder.layers[i] = self.layers[i]

        return latent_decoder


class Code(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.K = num_codes
        self.embedding = nn.Embedding(num_codes, code_dim, padding_idx=-1)
        self.embedding.weight.data.uniform_(-1. / num_codes, 1. / num_codes)

    def forward(self, indices=None):
        embed = self.embedding(indices)
        return embed

    def update_code(self, posterior):
        raise NotImplementedError


class EMACode(Code):
    def __init__(self, num_codes, code_dim, lamda=0.999, stop_gradient=False):
        super().__init__(num_codes, code_dim)
        self.lamda = lamda
        self.code_count = nn.Parameter(torch.zeros(num_codes).float(), requires_grad=False)
        self.update = not stop_gradient

    def forward(self, indices=None, inputs=None):
        if inputs is not None:
            return vq_search(inputs, self.embedding.weight)

        return super().forward(indices)

    def straight_through(self, z_e_x):
        z_st, indices = vq_st(z_e_x, self.embedding.weight.detach())
        z_bar = self.embedding.weight.index_select(dim=0, index=indices)
        z_bar = z_bar.view_as(z_e_x)
        return z_st, z_bar, indices.view(*z_st.size()[:-1])

    def update_code(self, posterior):
        z_enc = posterior['input'].view(-1, posterior['input'].size(-1))  # batch_size, sequence_length, D
        enc_sum = self._count_ema(z_enc, posterior["mask"], posterior["tgt"])
        self._code_ema(enc_sum)

    def _code_ema(self, z_repr):
        """ exponential moving average """
        count = self.code_count.view(self.K, -1)  # K,1
        mask = (count > 0.0).float()  # K,1
        code = self.embedding.weight.data
        code = mask * (code * self.lamda + (1 - self.lamda) * z_repr / (count + (1 - mask) * INF)) + (1 - mask) * code

        self.embedding.weight.data = code
        self.embedding.weight.requires_grad = self.update

    def _count_ema(self, enc, mask, idx):
        mask = mask.long()
        idx = idx * mask - (1 - mask)  # set the masked indices is -1

        enc = enc.view(-1, enc.size(-1))
        idx = idx.view(-1)
        z_exp = []
        for i in range(self.K):
            i_hit = idx == i  # batch_size*sequence_length,1
            self.code_count[i] = self.lamda * self.code_count[i] + i_hit.sum().float() * (1 - self.lamda)
            z_i_sum = enc[i_hit].sum(dim=0)
            z_exp.append(z_i_sum)

        return torch.stack(z_exp)


class Dictionary(object):
    # helper class for extend the NAT base
    def __init__(self, num_codes):
        super().__init__()
        self.num_codes = num_codes

    def bos(self):
        return -1

    def eos(self):
        return -1

    def unk(self):
        return -1

    def pad(self):
        return -1

    def __len__(self):
        return self.num_codes


def base_architecture(args):
    from latent_nat.awesome_nat import base_architecture
    base_architecture(args)


@register_model_architecture("cnat", "cnat_wmt14")
def cnat_wmt14(args):
    from latent_nat.awesome_nat import awesome_nat_wmt14
    awesome_nat_wmt14(args)
    base_architecture(args)


@register_model_architecture('cnat', 'cnat_iwslt16')
def cnat_iwslt16(args):
    from latent_nat.awesome_nat import awesome_nat_iwslt16
    awesome_nat_iwslt16(args)
    base_architecture(args)


@register_model_architecture('cnat', 'cnat_iwslt14')
def cnat_iwslt14(args):
    from latent_nat.awesome_nat import awesome_nat_iwslt14
    awesome_nat_iwslt14(args)
    base_architecture(args)


@register_model_architecture('cnat', 'cnat_base')
def cnat_base(args):
    from latent_nat.awesome_nat import awesome_nat_base
    awesome_nat_base(args)
    base_architecture(args)
