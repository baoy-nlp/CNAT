from collections import namedtuple

import torch
import torch.nn as nn

from .crf import crf_training, crf_inference, build_crf_layer
from .model_utils import VanillaTransformerDecoder, InputableLSTMDecoder, VanillaLSTMEncoder

PredictorOut = namedtuple("PredictorOut", ['out', 'token', "input"])


class NARPredictor(nn.Module):
    """ Actually, is the basic predictor """

    def __init__(self, args, padding_idx=-1, embed_positions=None, embed_tokens=None):
        super().__init__()
        decoder_cls = DECODER_FUNC_DICT[args.predictor_decoder_cls.upper()]
        self.decoder = decoder_cls(
            args,
            padding_idx=padding_idx,
            embed_positions=embed_positions if args.share_learned_pos else None,
            embed_tokens=embed_tokens
        )
        # do not set an extra encoder ?

    def forward(self, inputs, mask, encoder_out=None, **unused):
        model_out, loss_ret = self.decoder(inputs, mask, encoder_out, **unused)
        loss_ret["out"] = model_out.out
        return model_out, loss_ret

    @staticmethod
    def add_args(parser):
        parser.add_argument('--share-learned-pos', action='store_true')
        parser.add_argument("--predictor-decoder-cls", type=str, default="NAR")
        parser.add_argument("--predictor-encoder-layers", type=int, default=0)
        parser.add_argument("--predictor-layers", type=int, metavar='N')
        parser.add_argument('--predictor-learned-pos', action='store_true')
        parser.add_argument('--predictor-feed-size', type=int, metavar='N')
        parser.add_argument("--predictor-iteration", type=int, default=4)
        parser.add_argument("--iter-output-func", type=str, default='add')
        parser.add_argument("--iter-embed-func", type=str, default="None")

        parser.add_argument("--crf-cls", type=str, default="DCRF")
        parser.add_argument("--crf-lowrank-approx", type=int,
                            help="the dimension of low-rank approximation of transition")
        parser.add_argument("--crf-beam-approx", type=int,
                            help="the beam size for apporixmating the normalizing factor")
        parser.add_argument("--crf-num-head", type=int,
                            help="the beam size for apporixmating the normalizing factor")
        parser.add_argument("--crf-input-last", action="store_true", default=False,
                            help="use the encoded output as the crf layer inputs")
        parser.add_argument("--no-noise-input", action="store_true", default=False,
                            help="use the noise decoder inputs FOR VQ Predicting")
        parser.add_argument("--use-emission", action="store_true", default=False,
                            help="add supervision to the emission score distribution")
        parser.add_argument("--replace-output", action="store_true", default=False,
                            help="revise the sampling distribution of crf"
                            )


class NARDecoder(nn.Module):
    """
    modeling the quantized vector with non-autoregressive model
    """

    def __init__(self, args, padding_idx=-1, embed_positions=None, embed_tokens=None):
        super().__init__()
        self.decoder = VanillaTransformerDecoder(args, padding_idx, embed_positions)
        self.output_projection = nn.Linear(self.decoder.embed_dim, args.vq_num, bias=False)
        if embed_tokens is not None and args.share_decoder_input_output_embed:
            self.output_projection.weight = embed_tokens.weight
        self.remove_inputs = getattr(args, "no_noise_input", False)

    @classmethod
    def build_decoder(cls, args, padding_idx, embed_positions, embed_tokens):
        return cls(args, padding_idx, embed_positions, embed_tokens)

    def forward(self, inputs, decoder_padding_mask, encoder_out=None, tgt_tokens=None, **unused) -> (
            PredictorOut, dict):
        inputs, ret = self.decoder(
            inputs=None if self.remove_inputs else inputs,
            decoder_padding_mask=decoder_padding_mask,
            encoder_out=encoder_out
        )
        logits = self.output_projection(inputs)

        predict_out = PredictorOut(
            out=logits,
            token=logits.max(dim=-1)[1],
            input=inputs,
        )
        loss_ret = {}
        if tgt_tokens is not None:
            loss_ret["vq-L1"] = {"out": logits, "tgt": tgt_tokens, "mask": ~decoder_padding_mask, "factor": 1.0}
        return predict_out, loss_ret


class CRFDecoder(NARDecoder):
    """
    modeling the quantized vector with conditional random fields
    """

    def __init__(self, args, padding_idx=-1, embed_positions=None, embed_tokens=None):
        super().__init__(args, padding_idx, embed_positions, embed_tokens)
        self.crf_layer = build_crf_layer(args)
        self.input_last = getattr(args, "crf_input_last", False)
        self.use_emission = getattr(args, "use_emission", False)
        self.replace_output = getattr(args, "replace_output", False)

    def forward(self, inputs, decoder_padding_mask, encoder_out=None, tgt_tokens=None, **unused) -> (
            PredictorOut, dict):
        if self.training:
            assert tgt_tokens is not None, 'teacher forcing need target should not be None'
        predict_out, _ = super().forward(inputs, decoder_padding_mask, encoder_out)
        word_ins_out = predict_out.out
        word_ins_mask = ~decoder_padding_mask
        loss_ret, _tokens = {}, None

        if tgt_tokens is None:  # inference stage
            _scores, _tokens = crf_inference(
                crf_layer=self.crf_layer,
                inputs=inputs if not self.input_last else predict_out.input,
                word_ins_out=word_ins_out,
                word_ins_mask=word_ins_mask
            )

        else:  # training stage
            loss_ret = crf_training(
                crf_layer=self.crf_layer,
                inputs=inputs if not self.input_last else predict_out.input,
                word_ins_out=word_ins_out,
                tgt_tokens=tgt_tokens,
                word_ins_mask=word_ins_mask,
                log_prob=unused.get("log_prob", True),
                use_emission=self.use_emission
            )

            if self.replace_output:
                # correction for CRF
                word_ins_out = loss_ret['out']

        prior_out = PredictorOut(
            out=word_ins_out,
            token=_tokens,
            input=predict_out.input
        )
        return prior_out, loss_ret


class LSTMCRFDecoder(nn.Module):
    def __init__(self, args, padding_idx=-1, embed_positions=None, embed_tokens=None):
        super().__init__()
        self.bidir = getattr(args, "bidir", True)
        self.lstm_layer = VanillaLSTMEncoder(
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_embed_dim,
            num_layers=args.predictor_layers,
            dropout_in=args.dropout,
            dropout_out=args.dropout,
            bidirectional=self.bidir
        )
        if self.bidir:
            self.hid_to_out = nn.Linear(args.decoder_embed_dim * 2, args.decoder_embed_dim)
        self.output_projection = nn.Linear(args.decoder_embed_dim, args.vq_num, bias=False)
        if embed_tokens is not None and args.share_decoder_input_output_embed:
            self.output_projection.weight = embed_tokens.weight

        self.crf_input_last = getattr(args, "crf_input_last", False)
        self.crf_layer = build_crf_layer(args)

    def forward(self, inputs, decoder_padding_mask, encoder_out=None, tgt_tokens=None, **unused) -> (
            PredictorOut, dict):
        if self.training:
            assert tgt_tokens is not None, 'teacher forcing need target should not be None'

        encoded_output = self.lstm_layer(x=inputs, padding_mask=decoder_padding_mask, enforce_sorted=False)
        if self.bidir:
            decoder_in = self.hid_to_out(encoded_output[0]).transpose(0, 1)
        else:
            decoder_in = encoded_output[0].transpose(0, 1)

        word_ins_out = self.output_projection(decoder_in)
        word_ins_mask = ~decoder_padding_mask
        loss_ret = {}

        if tgt_tokens is None:
            # inference stage
            _scores, _tokens = crf_inference(
                crf_layer=self.crf_layer,
                inputs=decoder_in if self.crf_input_last else inputs,
                word_ins_out=word_ins_out,
                word_ins_mask=word_ins_mask
            )
            prior_out = PredictorOut(
                out=word_ins_out,
                token=_tokens,
                input=decoder_in
            )
        else:
            # training stage
            prior_out = PredictorOut(out=word_ins_out, token=None, input=decoder_in)
            loss_ret = crf_training(
                crf_layer=self.crf_layer,
                inputs=decoder_in if self.crf_input_last else inputs,
                word_ins_out=word_ins_out,
                tgt_tokens=tgt_tokens,
                word_ins_mask=word_ins_mask,
                log_prob=True if "log_prob" not in unused else unused["log_prob"]
            )

        return prior_out, loss_ret


class ARDecoder(nn.Module):
    def __init__(self, args, padding_idx=-1, embed_positions=None, embed_tokens=None):
        super().__init__()
        self.bos = args.vq_num
        self.pad = args.vq_num + 1
        num_embed = args.vq_num + 2
        pretrained_embed = extend_embedding(
            num_embeddings=num_embed,
            embed_dim=args.decoder_embed_dim,
            padding_idx=self.pad,
            embed_tokens=embed_tokens,
        )
        self.decoder = InputableLSTMDecoder(
            num_embeddings=num_embed,
            padding_idx=self.pad,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_ffn_embed_dim,
            out_embed_dim=args.decoder_embed_dim,
            num_layers=args.predictor_layers,
            dropout_in=args.dropout,
            dropout_out=args.dropout,
            attention=True,
            encoder_output_units=args.encoder_embed_dim,
            input_feed_size=getattr(args, "predictor_feed_size", args.decoder_embed_dim),
            pretrained_embed=pretrained_embed,
            share_input_output_embed=getattr(args, "predictor_share_input_output_embed",
                                             args.share_decoder_input_output_embed),
            adaptive_softmax_cutoff=getattr(args, "predictor_adaptive_softmax_cutoff", args.adaptive_softmax_cutoff),
            max_target_positions=args.max_target_positions,
            residuals=False
        )

    def init_prev_output_tokens(self, inputs, tgt_tokens: torch.Tensor):
        bos_tokens = inputs.new_zeros(inputs.size(0), 1).long() + self.bos
        if tgt_tokens is not None:
            return torch.cat((bos_tokens, tgt_tokens[:, :-1]), dim=-1)
        else:
            return bos_tokens

    def forward(self, inputs, decoder_padding_mask, encoder_out=None, tgt_tokens=None, **unused) -> (
            PredictorOut, dict):
        if self.training:
            assert tgt_tokens is not None, 'teacher forcing need target should not be None'
        prev_output_tokens = self.init_prev_output_tokens(inputs, tgt_tokens=tgt_tokens)
        out, attn_scores = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            inputs=inputs
        )
        logits = out.contiguous()[:, :, :self.bos]
        predict_out = PredictorOut(
            out=logits,
            token=logits.max(dim=-1)[1],
            input=inputs
        )
        loss_ret = {}
        if tgt_tokens is not None:
            loss_ret["vq-L1"] = {
                "out": out,
                "tgt": tgt_tokens,
                "mask": ~decoder_padding_mask,
                "factor": 1.0
            }
        return predict_out, loss_ret


class IRDecoder(NARDecoder):
    def __init__(self, args, padding_idx=-1, embed_positions=None, embed_tokens=None):
        super().__init__(args, padding_idx, embed_positions, embed_tokens)
        self.embed_tokens = embed_tokens

        self.max_iter = getattr(args, "predict_iteration", 4)
        self.iter_output_func = getattr(args, "iter_output_func", "add")
        self.iter_embed_func = getattr(args, "iter_embed_func", None)

        # define the iterative refine decoder
        self.iter_decoder = VanillaTransformerDecoder(args, padding_idx, embed_positions)

    def forward_nar_prediction(self, outputs, decoder_padding_mask, encoder_out=None, **unused):
        outputs, ret = self.decoder.forward(outputs, decoder_padding_mask, encoder_out)
        logits = self.output_projection(outputs)
        return logits, ret

    def forward_embedding(self, logits):
        if self.iter_embed_func == "exp":
            weights = self.embed_tokens.weight  # num_embeddings, dim
            return logits @ weights

        indices = logits.max(-1)[1]
        return self.embed_tokens(indices)

    def forward_decoder_inputs(self, inputs, logits, **unused):
        if self.iter_output_func == "add":
            decoder_inputs = inputs + self.forward_embedding(logits)
        elif self.iter_output_func == "replace":
            decoder_inputs = self.forward_embedding(logits)
        else:
            decoder_inputs = self.forward_embedding(logits)
        return decoder_inputs

    def forward_iter_prediction(self, inputs, logits, decoder_padding_mask, encoder_out=None, **unused):
        logits_ret = [logits, ]
        for it in range(1, self.max_iter):
            decoder_inputs = self.forward_decoder_inputs(inputs=inputs, logits=logits_ret[-1])
            outputs, ret = self.iter_decoder(
                inputs=decoder_inputs,
                decoder_padding_mask=decoder_padding_mask,
                encoder_out=encoder_out
            )
            _logits = self.output_projection(outputs)
            logits_ret.append(_logits)
        return logits_ret

    def forward(self, inputs, decoder_padding_mask, encoder_out=None, **unused):
        logits, ret = self.forward_nar_prediction(inputs, decoder_padding_mask, encoder_out)
        logits_list = self.forward_iter_prediction(inputs, logits, decoder_padding_mask, encoder_out, **unused)
        return logits_list, ret


def extend_embedding(num_embeddings, embed_dim, padding_idx=-1, embed_tokens=None):
    from fairseq.models.lstm import Embedding
    m = Embedding(num_embeddings, embedding_dim=embed_dim, padding_idx=padding_idx)
    if embed_tokens is not None:
        num_size = embed_tokens.weight.data.size(0)
        m.weight.data[:num_size, :] = embed_tokens.weight.data
    return m


DECODER_FUNC_DICT = {
    "AR": ARDecoder,
    "NAR": NARDecoder,
    "IR": IRDecoder,
    "CRF": CRFDecoder,
    "LSTMCRF": LSTMCRFDecoder
}
