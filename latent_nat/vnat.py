import copy

import torch
import torch.nn as nn
from fairseq.models import register_model, register_model_architecture

from .glat import GlancingTransformer, GlancingTransformerDecoder, init_bert_params
from .utils import GateNet, SelfATTEncoder, GaussianVariable, GlobalNames

try:
    from fairseq.models.transformer import EncoderOut
except ImportError:
    from fairseq.models.fairseq_encoder import EncoderOut


@register_model("vnat")
class VariationalNAT(GlancingTransformer):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = VNATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @staticmethod
    def add_args(parser):
        GlancingTransformer.add_args(parser)
        VNATDecoder.add_args(parser)

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
            latent_factor = getattr(self.args, "latent_factor", 1.0)
            losses["KL"] = {
                "loss": self._compute_latent_loss(
                    inner_states[GlobalNames.PRI_RET],
                    inner_states[GlobalNames.POST_RET]
                ) * latent_factor,
                "factor": latent_factor
            }

        return losses

    @classmethod
    def _compute_latent_loss(cls, prior_out, posterior_out):
        # prior
        mean1 = prior_out[GlobalNames.MEAN]
        logv1 = prior_out[GlobalNames.LOGV]
        var1 = logv1.exp()

        mean2 = posterior_out[GlobalNames.MEAN]
        logv2 = posterior_out[GlobalNames.LOGV]
        var2 = logv2.exp()

        kl = 0.5 * (logv2 - logv1 + (var1 / var2) + (mean2 - mean1).pow(2) / var2 - 1).sum(dim=-1).mean()
        return kl


class VNATDecoder(GlancingTransformerDecoder):
    """
    Extra attributions:
        - posterior
        - prior
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.posterior = self._build_posterior()
        self.prior = self._build_prior()
        self.gate = GateNet(
            d_model=self.embed_dim * 2,
            d_hidden=self.embed_dim * 4,
            d_output=1 if getattr(args, "use_scalar_gate", True) else self.embed_dim,
            dropout=args.dropout
        ) if args.combine_func == "residual" else None

    @staticmethod
    def add_args(parser):
        parser.add_argument("--latent-factor", type=float, default=1.0)
        parser.add_argument("--latent-dim", type=int, default=200)
        parser.add_argument("--latent-layers", type=int, default=5)
        parser.add_argument("--combine-func", type=str, default="residual")
        parser.add_argument("--use-scalar-gate", action="store_true", default=False)

    def _extract_features(
            self,
            inputs,
            decoder_padding_mask,
            pos=None,
            encoder_out=None,
            early_exit=None,
            tgt_tokens=None,
            **unused,
    ):
        if tgt_tokens is not None and self.glat_training and self.training:
            # Glancing Training
            z, z_ret = self.forward_latent(encoder_out, tgt_tokens, inputs, decoder_padding_mask=decoder_padding_mask)

            # integrating the latent variable information
            feat = self.forward_combine(inputs, z)

            # first decoding pass
            outputs, first_pass_ret = self._forward_decoding(feat, decoder_padding_mask, encoder_out, early_exit)

            # glancing for second pass
            glancing_inputs, predict, glancing_mask = self.glancing(
                features=outputs, targets=tgt_tokens, mask=decoder_padding_mask, ratio=self.sampling_ratio,
                inputs=inputs
            )

            # integrating the latent variable information
            feat = self.forward_combine(glancing_inputs, z)

            # second decoding pass
            features, ret = self._forward_decoding(feat, decoder_padding_mask, encoder_out, early_exit)

            ret[GlobalNames.INPUT] = inputs
            ret[GlobalNames.FEATURES] = outputs
            ret[GlobalNames.PREDICTS] = predict
            ret[GlobalNames.GLANCING_INPUTS] = glancing_inputs
            ret[GlobalNames.GLANCING_MASK] = glancing_mask

            ret.update(z_ret)
            return features, ret
        else:
            z, z_ret = self.forward_latent(encoder_out, tgt_tokens, inputs, decoder_padding_mask=decoder_padding_mask)
            feats = self.forward_combine(inputs, z)
            # decoding only one pass during inference
            features, ret = self._forward_decoding(feats, decoder_padding_mask, encoder_out, early_exit)
            ret[GlobalNames.INPUT] = inputs
            ret[GlobalNames.FEATURES] = features
            ret.update(z_ret)
            return features, ret

    def forward_latent(self, encoder_out: EncoderOut, tgt_tokens=None, inputs=None, **unused):
        prior_out = self.prior.forward(inputs=encoder_out.encoder_out, mask=~encoder_out.encoder_padding_mask)
        inner_states = {GlobalNames.PRI_RET: prior_out}

        z = prior_out[GlobalNames.REC]  # batch_size, hidden
        if tgt_tokens is not None:
            y_mask = tgt_tokens.ne(self.padding_idx)
            y_embed = self.forward_embedding(tgt_tokens)[0]
            posterior_out = self.posterior(
                x_embed=encoder_out.encoder_out,
                y_embed=y_embed,
                x_mask=~encoder_out.encoder_padding_mask,
                y_mask=y_mask
            )
            inner_states[GlobalNames.POST_RET] = posterior_out

            z = posterior_out[GlobalNames.REC]
        z = z.unsqueeze(1).contiguous().expand(-1, inputs.size(1), -1)
        return z, inner_states

    def forward_combine(self, inputs, z):
        if self.gate is not None:
            g = self.gate(torch.cat([inputs, z], dim=-1)).sigmoid()
            inputs = inputs * g + z * (1 - g)
        else:
            inputs = inputs + z
        return inputs

    def _build_posterior(self):
        model_args = self.args
        args = copy.deepcopy(model_args)
        args.encoder_layers = getattr(model_args, "latent_layers", model_args.decoder_layers)
        return VAEPosterior(args)

    def _build_prior(self):
        model_args = self.args
        args = copy.deepcopy(model_args)
        return VAEPrior(args)


class VAEPrior(nn.Module):
    """
        p(z|x): mapping enc(x) to mean and logv
    """

    def __init__(self, args):
        super().__init__()
        self.latent = GaussianVariable(
            input_dim=args.encoder_embed_dim,
            latent_dim=getattr(args, "latent_dim", 200),
            output_dim=args.encoder_embed_dim
        )

    def forward(self, inputs, mask=None):
        inputs = inputs.transpose(0, 1)
        if mask is not None:
            h_f = (inputs * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=-1).float().unsqueeze(-1)
        else:
            h_f = inputs.mean(dim=1)

        return self.latent.forward(inputs=h_f)


class VAEPosterior(nn.Module):
    """
        q(z|x,y): enc(y) and enc(x), mapping enc(x,y) to mean and logv
    """

    def __init__(self, args):
        super().__init__()

        self.y_encoder = SelfATTEncoder(args)

        self.latent = GaussianVariable(
            input_dim=args.encoder_embed_dim * 2,
            latent_dim=getattr(args, "latent_dim", 200),
            output_dim=args.encoder_embed_dim
        )

    def forward(self, x_embed, y_embed, x_mask=None, y_mask=None):
        def _compute_inputs(inputs, mask=None):
            if mask is not None:
                _h = (inputs * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=-1).float().unsqueeze(-1)
            else:
                _h = inputs.mean(dim=1)
            return _h

        x_output = x_embed.transpose(0, 1)
        h_f = _compute_inputs(x_output, x_mask)

        # encoding y
        y_output = self.y_encoder.forward(y_embed, ~y_mask).encoder_out
        y_output = y_output.transpose(0, 1)
        h_e = _compute_inputs(y_output, y_mask)

        # concatenate x and y
        h = torch.cat([h_f, h_e], dim=-1)
        return self.latent.forward(inputs=h)


def base_architecture(args):
    from nat_base.vanilla_nat import base_architecture
    base_architecture(args)


@register_model_architecture("vnat", "vnat_wmt14")
def vnat_wmt14(args):
    from latent_nat.glat import glat_wmt14
    glat_wmt14(args)
    base_architecture(args)


@register_model_architecture('vnat', 'vnat_iwslt16')
def vnat_iwslt16(args):
    from latent_nat.glat import glat_iwslt16
    glat_iwslt16(args)
    base_architecture(args)


@register_model_architecture('vnat', 'vnat_iwslt14')
def vnat_iwslt14(args):
    from latent_nat.glat import glat_iwslt14
    glat_iwslt14(args)
    base_architecture(args)


@register_model_architecture('vnat', 'vnat_base')
def vnat_base(args):
    from latent_nat.glat import glat_base
    glat_base(args)
    base_architecture(args)
