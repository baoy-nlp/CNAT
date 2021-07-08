from collections import namedtuple

import torch.nn as nn

from latent_nat.awesome_nat import NATDecoder
from latent_nat.crf import crf_inference, crf_training, CRF_CLS
from latent_nat.modules import TransformerDecoderFunc

INF: float = 1e-10

DecoderOut = namedtuple("DecoderOut", ['out', 'token', "feature"])


class CRFPredictor(NATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        delattr(self, "embed_length")  # We do not need length prediction
        self.crf = CRF_CLS[getattr(args, "crf_cls", "DCRF")](
            num_embedding=args.num_codes,
            low_rank=getattr(args, "crf_lowrank_approx", 32),
            beam_size=getattr(args, "crf_beam_approx", 64),
            num_head=getattr(args, "crf_num_head", 8),
            embed_size=args.decoder_embed_dim
        )

        self.use_emission = getattr(args, "use_emission", False)

    def preprocess(self, inputs, decoder_padding_mask, include_pos=False):
        if include_pos or self.args.self_attn_cls == "shaw":
            return inputs
        if self.padding_idx is not None:
            mask = decoder_padding_mask.long()
            pos_tokens = (mask * self.padding_idx) + (1 - mask) * (self.padding_idx + 1)
        else:
            pos_tokens = (~decoder_padding_mask).long()
        inputs = self.forward_embedding(
            prev_output_tokens=pos_tokens,
            states=inputs,
            add_position=True
        )[0]
        return inputs

    def forward(self, inputs, decoder_padding_mask, encoder_out=None, tgt_tokens=None, **unused):
        inputs = self.preprocess(
            inputs=inputs, decoder_padding_mask=decoder_padding_mask, include_pos=unused.get("include_pos", False)
        )
        features, _ = self._extract_features(
            x=inputs,
            decoder_padding_mask=decoder_padding_mask,
            encoder_out=encoder_out,
            **unused
        )
        logits = self.output_layer(features)

        ret, _tokens = {}, None

        if tgt_tokens is None:
            # inference stage
            _scores, _tokens = crf_inference(
                crf_layer=self.crf,
                inputs=features,
                emission=logits,
                forward_mask=~decoder_padding_mask
            )

        else:
            # training stage
            ret = crf_training(
                crf_layer=self.crf,
                inputs=features,
                emission=logits,
                tgt_tokens=tgt_tokens,
                forward_mask=~decoder_padding_mask,
                log_prob=unused.get("log_prob", True),
                include_emission_loss=self.use_emission
            )
            logits = ret['out']
            _tokens = ret["out"].max(dim=-1)[1]
            ret["VQ"] = ret["CRF"]
            if "CRF-emission" in ret:
                ret["VQ-Out"] = ret["CRF-emission"]

        outputs = DecoderOut(
            out=logits,
            token=_tokens,
            feature=features
        )

        return outputs, ret


class CRFPredictor2(TransformerDecoderFunc):
    # Transformer Decoder Function + CRF Inference Layers
    def __init__(self, args, dictionary, embed_tokens=None, no_encoder_attn=False):
        super().__init__(args, padding_idx=-1, embed_positions=None)
        self.args = args
        self.output_projection = nn.Linear(self.embed_dim, dictionary.num_codes, bias=False)
        if embed_tokens is not None:
            self.output_projection.weight = embed_tokens.weight

        self.crf = CRF_CLS[getattr(args, "crf_cls", "DCRF")](
            num_embedding=args.num_codes,
            low_rank=getattr(args, "crf_lowrank_approx", 32),
            beam_size=getattr(args, "crf_beam_approx", 64),
            num_head=getattr(args, "crf_num_head", 8),
            embed_size=args.decoder_embed_dim
        )

        self.use_emission = getattr(args, "use_emission", False)

    def preprocess(self, inputs, decoder_padding_mask, include_pos=False):
        if include_pos or self.args.self_attn_cls == "shaw":
            return inputs
        mask = decoder_padding_mask.long()
        pos_tokens = (mask * self.padding_idx) + (1 - mask) * (self.padding_idx + 1)
        inputs = self.forward_embedding(inputs, pos_tokens)[0]
        return inputs

    def forward(self, inputs, decoder_padding_mask, encoder_out=None, tgt_tokens=None, **unused):
        inputs = self.preprocess(
            inputs=inputs, decoder_padding_mask=decoder_padding_mask, include_pos=unused.get("include_pos", False)
        )
        features, _ = self._extract_features(
            inputs,
            decoder_padding_mask=decoder_padding_mask,
            encoder_out=encoder_out,
        )
        # emission score
        logits = self.output_projection(features)

        ret, _tokens = {}, None

        if tgt_tokens is None:
            # final score and its target tokens
            _scores, _tokens = crf_inference(
                crf_layer=self.crf,
                inputs=features,
                emission=logits,
                forward_mask=~decoder_padding_mask
            )

        else:
            ret = crf_training(
                crf_layer=self.crf,
                inputs=features,
                emission=logits,
                tgt_tokens=tgt_tokens,
                forward_mask=~decoder_padding_mask,
                log_prob=unused.get("log_prob", True),
                include_emission_loss=self.use_emission
            )
            logits = ret['out']
            _tokens = ret["out"].max(dim=-1)[1]
            ret["VQ"] = ret["CRF"]
            if "CRF-emission" in ret:
                ret["VQ-Out"] = ret["CRF-emission"]

        outputs = DecoderOut(
            out=logits,
            token=_tokens,
            feature=features
        )

        return outputs, ret
