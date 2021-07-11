import random

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from latent_nat.awesome_nat import AwesomeNAT, NATDecoder, ensemble_decoder
from latent_nat.utils import GlobalNames


@register_model("glat")
class GlancingTransformer(AwesomeNAT):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = GlancingTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @staticmethod
    def add_args(parser):
        AwesomeNAT.add_args(parser)
        GlancingTransformerDecoder.add_args(parser)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        word_ins_out, inner_states = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
            extra_ret=True,
        )
        return self._compute_loss(word_ins_out, tgt_tokens, encoder_out, inner_states)

    def model_step_update(self, step_num):
        """ interface for applying the step schedule """
        self.decoder.step(step_num)

    def _compute_loss(self, word_ins_out, tgt_tokens, encoder_out, inner_states=None):
        word_ins_ret = self._compute_glancing_loss(word_ins_out, tgt_tokens, inner_states)
        word_ins_ret["ls"] = self.args.label_smoothing

        losses = {"word_ins": word_ins_ret}
        losses.update(self._compute_length_loss(encoder_out=encoder_out, tgt_tokens=tgt_tokens))

        return losses

    def _compute_glancing_loss(self, decode_out, tgt_tokens, inner_states=None, mask=None):
        mask = tgt_tokens.ne(self.pad) if mask is None else mask
        if inner_states is not None and GlobalNames.GLANCING_MASK in inner_states:
            mask = (inner_states[GlobalNames.GLANCING_MASK].squeeze(-1) < 1.0) * mask
        return {
            "out": decode_out,
            "tgt": tgt_tokens,
            "mask": mask,
            "nll_loss": True
        }


class GlancingTransformerDecoder(NATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        # hyper-parameter
        self.teaching_mode = args.teaching_mode
        self.glat_training = getattr(self, "teaching_mode", "schedule") != "schedule"
        self.y_sampler = ReferenceSampler(num_mode=args.glancing_num_mode, sub_mode=args.glancing_sub_mode)
        self.y_ratio_scheduler = StepAnnealScheduler(args)

    @staticmethod
    def add_args(parser, key=None):
        if key is None:
            parser.add_argument("--teaching-mode", type=str, choices=["glancing", "schedule"], default=None)
            parser.add_argument("--glancing-num-mode", type=str,
                                choices=["fixed", "adaptive", "adaptive-uni", "adaptive-rev"],
                                default="adaptive", help="glancing sampling number")
            parser.add_argument("--glancing-sub-mode", type=str, choices=["mixing", "schedule"],
                                default="mixing", help="uniform: mixing the decoder inputs and oracle, "
                                                       "schedule: mixing the predict and oracle")
            # step annealing scheduler
            StepAnnealScheduler.add_args(parser)
        else:
            parser.add_argument("--{}-teaching-mode".format(key), type=str, choices=["glancing", "schedule"],
                                default=None)
            parser.add_argument("--{}-glancing-num-mode".format(key), type=str,
                                choices=["fixed", "adaptive", "adaptive-uni", "adaptive-rev"],
                                default="adaptive", help="glancing sampling number")
            parser.add_argument("--{}-glancing-sub-mode".format(key), type=str, choices=["mixing", "schedule"],
                                default="mixing", help="uniform: mixing the decoder inputs and oracle, "
                                                       "schedule: mixing the predict and oracle")
            StepAnnealScheduler.add_args(parser, key)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, tgt_tokens=None, **unused):
        features, ret = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            tgt_tokens=tgt_tokens,
            **unused
        )
        decoder_out = self.output_layer(features)
        decoder_out = F.log_softmax(decoder_out, -1) if normalize else decoder_out

        if unused.get("extra_ret", False) and tgt_tokens is not None:
            return decoder_out, ret
        else:
            return decoder_out

    def glancing(self, features, targets, mask, ratio=0.5, inputs=None, **kwargs):
        """ sampling the reference and mixed the inputs"""
        logits = self.output_layer(features)
        prob, predict = logits.max(dim=-1)
        pred_embed = self.forward_embedding(predict)[0]

        sample = self.y_sampler.forward_sampling(targets=targets, padding_mask=mask, ratio=ratio, logits=logits)
        observed = sample.float().unsqueeze(-1)
        ref_embed = self.forward_embedding(targets)[0]

        decode_inputs = self.y_sampler.forward_inputs(
            inputs=inputs, ref=ref_embed, observed=observed, pred=pred_embed, s_mode=kwargs.get("s_mode", None)
        )
        return decode_inputs, predict, observed

    def decode(
            self,
            x,
            decoder_padding_mask,
            pos=None,
            encoder_out=None,
            early_exit=None,
            tgt_tokens=None,
            **unused
    ):
        """ including two decoding passes while training with glancing sampling """
        if tgt_tokens is not None and self.glat_training and self.training:
            # Glancing Training
            # first decoding pass
            with torch.no_grad():
                outputs, first_pass_ret = self._forward_decoding(x, decoder_padding_mask, encoder_out, early_exit)

            # glancing for second pass
            glancing_inputs, predict, glancing_mask = self.glancing(
                features=outputs, targets=tgt_tokens, mask=decoder_padding_mask, ratio=self.sampling_ratio, inputs=x
            )

            # second decoding pass
            features, ret = self._forward_decoding(glancing_inputs, decoder_padding_mask, encoder_out, early_exit)
            ret[GlobalNames.INPUT] = x
            ret[GlobalNames.FEATURES] = outputs
            ret[GlobalNames.PREDICTS] = predict
            ret[GlobalNames.GLANCING_INPUTS] = glancing_inputs
            ret[GlobalNames.GLANCING_MASK] = glancing_mask
            return features, ret
        else:
            # decoding only one pass during inference
            features, ret = self._forward_decoding(x, decoder_padding_mask, encoder_out, early_exit)
            ret[GlobalNames.INPUT] = x
            ret[GlobalNames.FEATURES] = features
            ret[GlobalNames.GLANCING_MASK] = decoder_padding_mask
            return features, ret

    def _forward_decoding(self, x, decoder_padding_mask, encoder_out=None, early_exit=None):
        """ One decoding pass with x & enc outputs
         Transformer decoding function: computing hidden states given encoder outputs and decoder inputs """
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        for i, layer in enumerate(self.layers):
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if not self.layerwise_attn else encoder_out.encoder_states[i],
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

        return x, {"attn": attn, "inner_states": inner_states}

    def step(self, step_num):
        """ update the glancing ratio """
        self.y_ratio_scheduler.forward(step_num)

    @property
    def sampling_ratio(self):
        return self.y_ratio_scheduler.ratio


class StepAnnealScheduler(object):
    """
    Annealing for glancing ratio
    """

    def __init__(self, args, key=""):
        super().__init__()
        self.start_ratio = getattr(args, "{}_start_ratio".format(key), args.start_ratio)
        self.end_ratio = getattr(args, "{}_end_ratio".format(key), args.end_ratio)
        self.anneal_steps = getattr(args, "{}_anneal_steps".format(key), args.anneal_steps)
        self.anneal_start = getattr(args, "{}_anneal_start".format(key), args.anneal_start)

        self.anneal_end = self.anneal_start + self.anneal_steps
        self.step_ratio = (self.end_ratio - self.start_ratio) / self.anneal_steps

        self._ratio = self.start_ratio

    @staticmethod
    def add_args(parser, key=None):
        # step annealing scheduler
        if key is None or len(key) < 1:
            parser.add_argument("--start-ratio", type=float, default=0.5)
            parser.add_argument("--end-ratio", type=float, default=0.5)
            parser.add_argument("--anneal-steps", type=int, default=1)
            parser.add_argument("--anneal-start", type=int, default=300000)
        else:
            parser.add_argument("--{}-start-ratio".format(key), type=float, default=0.5)
            parser.add_argument("--{}-end-ratio".format(key), type=float, default=0.5)
            parser.add_argument("--{}-anneal-steps".format(key), type=int, default=1)
            parser.add_argument("--{}-anneal-start".format(key), type=int, default=300000)

    def forward(self, step_num):
        if step_num < self.anneal_start:
            return self.start_ratio
        elif step_num >= self.anneal_end:
            return self.end_ratio
        else:
            self._ratio = self.start_ratio + self.step_ratio * (step_num - self.anneal_start)
            return self._ratio

    @property
    def ratio(self):
        return self._ratio


class ReferenceSampler(object):
    def __init__(self, num_mode, sub_mode):
        super().__init__()
        self.num_mode = num_mode  # compute substitution number
        self.sub_mode = sub_mode  # substitution mode

    def forward_sampling(self, targets, padding_mask, ratio=0.5, logits=None, n_mode=None, s_mode=None):
        return glancing_sampling(
            targets=targets, padding_mask=padding_mask, ratio=ratio, logits=logits,
            n_mode=self.num_mode if n_mode is None else n_mode,
            s_mode=self.sub_mode if s_mode is None else s_mode
        )

    def forward_inputs(self, inputs, ref, observed, pred=None, s_mode=None):
        s_mode = self.sub_mode if s_mode is None else s_mode

        if s_mode == "schedule":
            assert pred is not None, "schedule needs prediction"
            inputs = pred
        return (1 - observed) * inputs + observed * ref


def glancing_sampling(targets, padding_mask, ratio=0.5, logits=None, n_mode="adaptive", s_mode="mixing"):
    """return the positions to be replaced """
    if n_mode == "fixed":
        number = targets.size(1) * ratio + 1
    elif n_mode == "adaptive":
        # E * f_ratio: Qian et al. ACL 2021
        assert logits is not None, "logits should not be None"
        predict = logits.max(dim=-1)[1]
        distance = (predict.ne(targets) * ~padding_mask).float().sum(dim=-1)
        number = distance * ratio + 1
    elif n_mode == "adaptive-uni":
        # E * random ratio: Uniform sampling ratio for the model.
        assert logits is not None, "logits should not be None"
        ratio = random.random()
        predict = logits.max(dim=-1)[1]
        distance = (predict.ne(targets) * ~padding_mask).float().sum(dim=-1)
        number = distance * ratio + 1
    elif n_mode == "adaptive-rev":
        # E * (1-E/N): The more predicting error, the more sampling token
        predict = logits.max(dim=-1)[1]
        distance = (predict.ne(targets) * ~padding_mask).float().sum(dim=-1)
        ratio = 1.0 - distance / ((~padding_mask).float())
        number = distance * ratio + 1
    else:
        number = None

    score = targets.clone().float().uniform_()

    if s_mode == "mixing":
        # select replaced token from uniform distributions
        assert number is not None, "number should be decided before sampling"
        score.masked_fill_(padding_mask, 2.0)
        rank = score.sort(1)[1]
        cutoff = utils.new_arange(rank) < number[:, None].long()
        sample = cutoff.scatter(1, rank, cutoff)  # batch_size, sequence_length
    elif s_mode == "schedule":
        # select the replaced token with its modeled y probability
        assert logits is not None, "logits should not be None"
        prob = logits.softmax(dim=-1)
        ref_score = prob.view(-1, targets.size(-1)).contiguous().gather(1, targets.view(-1, 1)).view(*targets.size())
        sample = score.lt(ref_score) * (~padding_mask)
    else:
        raise RuntimeWarning("sample is none")

    return sample


def base_architecture(args):
    from latent_nat.awesome_nat import base_architecture
    base_architecture(args)


@register_model_architecture("glat", "glat_wmt14")
def glat_wmt14(args):
    from latent_nat.awesome_nat import awesome_nat_wmt14
    awesome_nat_wmt14(args)
    base_architecture(args)


@register_model_architecture('glat', 'glat_iwslt16')
def glat_iwslt16(args):
    from latent_nat.awesome_nat import awesome_nat_iwslt16
    awesome_nat_iwslt16(args)
    base_architecture(args)


@register_model_architecture('glat', 'glat_iwslt14')
def glat_iwslt14(args):
    from latent_nat.awesome_nat import awesome_nat_iwslt14
    awesome_nat_iwslt14(args)
    base_architecture(args)


@register_model_architecture('glat', 'glat_base')
def glat_base(args):
    from latent_nat.awesome_nat import awesome_nat_base
    awesome_nat_base(args)
    base_architecture(args)
