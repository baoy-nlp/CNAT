import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.nat.nonautoregressive_transformer import (
    DecoderOut,
    NATransformerModel,
    NATransformerDecoder,
    ensemble_decoder,
    init_bert_params,
    register_model_architecture,
    register_model,
    utils
)
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import LayerDropModuleList

from .layer import BlockedDecoderLayer, BlockedEncoderLayer
from .modules import RelativePositionEmbeddings

INF = 1e10

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


def build_relative_embeddings(args, embedding_dim=None):
    if embedding_dim is None:
        embedding_dim = args.decoder_embed_dim // getattr(args, "decoder_attention_heads")
    return RelativePositionEmbeddings(
        max_rel_positions=getattr(args, "max_rel_positions", 4),
        embedding_dim=embedding_dim,
        direction=True,
        dropout=args.dropout
    )


def _softcopy_assignment(src_lens, trg_lens, tau=0.3):
    max_trg_len = trg_lens.max()
    max_src_len = src_lens.max()
    index_s = utils.new_arange(src_lens, max_src_len).float()
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    diff = -(index_t[:, None] - index_s[None, :]).abs()  # max_trg_len, max_src_len
    diff = diff.unsqueeze(0).expand(trg_lens.size(0), *diff.size())
    mask = (src_lens[:, None] - 1 - index_s[None, :]).lt(0).float()  # batch_size, max_src_lens
    logits = (diff / tau - INF * mask[:, None, :])
    prob = logits.softmax(-1)
    return prob


def _interpolate_assignment(src_masks, tgt_masks, tau=0.3):
    max_src_len = src_masks.size(1)
    max_tgt_len = tgt_masks.size(1)
    src_lens = src_masks.sum(-1).float()
    tgt_lens = tgt_masks.sum(-1).float()
    index_t = utils.new_arange(tgt_masks, max_tgt_len).float()
    index_s = utils.new_arange(tgt_masks, max_src_len).float()
    steps = src_lens / tgt_lens
    index_t = steps[:, None] @ index_t[None, :]  # batch x max_trg_len
    index = (index_s[None, None, :] - index_t[:, :, None]) ** 2
    index = (-index.float() / tau - INF * (1 - src_masks[:, None, :].float())).softmax(dim=-1)
    return index


def load_pretrained_model(args):
    if getattr(args, "ptrn_model_path", None) is None:
        return None

    from fairseq import checkpoint_utils
    models, _ = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.ptrn_model_path),
        task=None,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    return models[0]


@register_model("awesome_nat")
class AwesomeNAT(NATransformerModel):
    """
    include the implementation of relative-position based model
    """

    @staticmethod
    def add_args(parser):
        # extend NATransformerModel with specific parameters
        NATransformerModel.add_args(parser)
        parser.add_argument(
            "--use-ptrn-encoder",
            action="store_true",
            help="whether use pretrained encoder"
        )
        parser.add_argument(
            "--use-ptrn-decoder",
            action="store_true",
            help="whether use pretrained decoder"
        )
        parser.add_argument(
            "--use-ptrn-embed",
            action="store_true",
            help="whether use pretrained embeddings"
        )
        parser.add_argument(
            "--fix-encoder",
            action="store_true",
            default=False
        )
        parser.add_argument(
            "--fix-decoder",
            action="store_true",
            default=False
        )
        parser.add_argument(
            "--fix-embed",
            action="store_true",
            default=False
        )
        parser.add_argument(
            "--self-attn-cls",
            type=str,
            default="abs",
            help="default 'abs' means attention with absolute position, otherwise, "
                 "we'll have a relative position based Attention Layer"
        )
        parser.add_argument(
            "--block-cls",
            type=str,
            default="residual",
            help="default is residual connection, otherwise, implement it with highway connection"
        )
        parser.add_argument(
            "--enc-block-cls",
            type=str,
            default="abs"
        )
        parser.add_argument(
            "--enc-self-attn-cls",
            type=str,
            default="abs"
        )
        parser.add_argument(
            "--dec-block-cls",
            type=str,
            default="abs"
        )
        parser.add_argument(
            "--dec-self-attn-cls",
            type=str,
            default="abs"
        )
        parser.add_argument(
            "--max-rel-positions",
            type=int,
            default=4,
            help="used in attention layer with relative positions"
        )
        parser.add_argument(
            "--share-rel-embeddings",
            action='store_true',
            help="whether share the representations across layers"
        )
        parser.add_argument(
            "--layer-norm-eps",
            type=float,
            default=1e-5
        )
        parser.add_argument(
            "--no-share-dec-input-output",
            action='store_true'
        )

        NATDecoder.add_args(parser)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=True, **kwargs)

        # decoding
        word_ins_out, inner_states = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            extra_ret=True
        )

        return self._compute_loss(word_ins_out, tgt_tokens, encoder_out)

    def _compute_loss(self, word_ins_out, tgt_tokens, encoder_out):
        losses = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True
            }
        }
        # length prediction
        losses.update(self._compute_length_loss(encoder_out=encoder_out, tgt_tokens=tgt_tokens))
        return losses

    def _compute_length_loss(self, encoder_out, tgt_tokens):
        ret = {}
        if self.decoder.length_loss_factor > 0:
            length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
            ret["length"] = {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }
        return ret

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        ptrn_model = load_pretrained_model(args)
        if getattr(args, "use_ptrn_encoder", False) and ptrn_model is not None:
            encoder = ptrn_model.encoder
        elif getattr(args, "use_ptrn_embed", False) and ptrn_model is not None:
            encoder = cls.build_encoder(args, src_dict, ptrn_model.encoder.embed_tokens)
        else:
            encoder_embed_tokens = cls.build_embedding(args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError("--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim")
            if args.decoder_embed_path and (args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError("--share-all-embeddings not compatible with --decoder-embed-path")
            decoder_embed_tokens = encoder.embed_tokens
            args.share_decoder_input_output_embed = not getattr(args, "no_share_dec_input_output", False)
        else:
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        model = cls(args, encoder, decoder)

        # add fine-tuning mode
        if getattr(args, "finetune", False):
            assert ptrn_model is not None, 'ptrn model should not be None while fine tuning the model'
            model.load_state_dict(state_dict=ptrn_model.state_dict(), strict=False)
            model.finetune(args)

        return model

    def finetune(self, args):
        """ finetune mode for the model parameters """
        parameters = []
        if getattr(args, "finetune_length_pred", False):
            for p in self.decoder.embed_length.parameters():
                parameters.append(p)
        if not getattr(args, "fix_encoder", False):
            for p in self.encoder.parameters():
                parameters.append(p)
        if not getattr(args, "fix_decoder", False):
            for p in self.decoder.parameters():
                parameters.append(p)
        # TODO: fix "hard to control the shared parameters between Encoder and Decoder."

        for p in self.parameters():
            if p in parameters:
                p.requires_grad = True and p.requires_grad
            else:
                p.requires_grad = False

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if (
                getattr(args, "enc_block_cls", "None") == "highway"
                or getattr(args, "enc_self_attn_cls", "abs") != "abs"
                or getattr(args, "layer_norm_eps", 1e-5) != 1e-5
        ):
            encoder = NATEncoder(args, src_dict, embed_tokens)
        else:
            encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        extra_ret, tgt_tokens = kwargs.get("extra_ret", False), kwargs.get("tgt_tokens", None)
        outputs = self.decoder.forward(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
            **kwargs
        )
        out, inner_states = (outputs[0], outputs[1]) if extra_ret else (outputs, None)

        _scores, _tokens = out.max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        orig = decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )
        if extra_ret:
            return orig, inner_states
        else:
            return orig

    def initialize_beam_output_tokens(self, encoder_out, src_tokens, tgt_tokens=None, beam_size=1):
        if beam_size == 1:
            return self.initialize_output_tokens(encoder_out, src_tokens, tgt_tokens)

        length_out = self.decoder.forward_length(normalize=True, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens, beam_size=beam_size)

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0) * beam_size, max_length).fill_(self.pad)
        initial_output_tokens.masked_fill_(idx_length[None, :] < length_tgt[:, None], self.unk)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(*initial_output_tokens.size()).type_as(
            encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, tgt_tokens=None):
        # length prediction
        if isinstance(self.decoder, NATDecoder):
            length_tgt = self.decoder.forward_length_prediction(
                length_out=self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
                encoder_out=encoder_out,
                tgt_tokens=tgt_tokens,
                use_ref_len=True,
            )
        else:
            length_tgt = self.decoder.forward_length_prediction(
                length_out=self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
                encoder_out=encoder_out,
                tgt_tokens=tgt_tokens,
            )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = length_tgt[:, None] + utils.new_arange(length_tgt, 1, beam_size) - beam_size // 2
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores
        )


class NATEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "enc_self_attn_cls", "abs") != "abs":
            self.embed_positions = None  # remove absolute position if we use relative positions

            if getattr(args, "share_rel_embeddings", False):
                rel_keys = build_relative_embeddings(args)
                rel_vals = build_relative_embeddings(args)
            else:
                rel_keys, rel_vals = None, None
            if self.encoder_layerdrop > 0.0:
                self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
            else:
                self.layers = nn.ModuleList([])
            self.layers.extend(
                [
                    self.build_encoder_layer(args, rel_keys, rel_vals)
                    for _ in range(args.encoder_layers)
                ]
            )

    def build_encoder_layer(self, args, rel_keys=None, rel_vals=None):
        if getattr(args, "enc_self_attn_cls", "abs") == "abs":
            return BlockedEncoderLayer(args)
        else:
            return BlockedEncoderLayer(
                args,
                relative_keys=rel_keys if rel_keys is not None else build_relative_embeddings(args),
                relative_vals=rel_vals if rel_vals is not None else build_relative_embeddings(args),
            )


class NATDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        # decoder inputs
        self.map_func = getattr(args, "mapping_func", "uniform")
        self.map_use = getattr(args, "mapping_use", "embed")

        # attention selection
        self.layerwise_attn = args.layerwise_attn
        if getattr(args, "self_attn_cls", "abs") != "abs":
            self.embed_positions = None  # remove absolute position if we use relative positions

            rel_keys = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            rel_vals = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            if self.decoder_layerdrop > 0.0:
                self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
            else:
                self.layers = nn.ModuleList([])
            self.layers.extend(
                [
                    self.build_decoder_layer(args, no_encoder_attn, rel_keys, rel_vals)
                    for _ in range(args.decoder_layers)
                ]
            )

    @staticmethod
    def add_args(parser):
        # input function
        parser.add_argument("--mapping-func", type=str, choices=["soft", "uniform", "interpolate"])
        parser.add_argument("--mapping-use", type=str, choices=["embed", "output"])
        parser.add_argument("--layerwise-attn", action='store_true', default=False)
        parser.add_argument("--layer-aggregate-func", type=str, default="mean")

    def build_decoder_layer(self, args, no_encoder_attn=False, rel_keys=None, rel_vals=None):
        if getattr(args, "block_cls", "None") == "highway" or getattr(args, "self_attn_cls", "abs") != "abs":
            if getattr(args, "self_attn_cls", "abs") == "abs":
                return BlockedDecoderLayer(args, no_encoder_attn)
            else:
                return BlockedDecoderLayer(
                    args, no_encoder_attn=no_encoder_attn,
                    relative_keys=rel_keys if rel_keys is not None else build_relative_embeddings(args),
                    relative_vals=rel_vals if rel_vals is not None else build_relative_embeddings(args),
                )

        return super().build_decoder_layer(args, no_encoder_attn)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, extra_ret=False, **unused):
        features, inner_states = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        decoder_out = F.log_softmax(decoder_out, -1) if normalize else decoder_out
        if extra_ret:
            return decoder_out, inner_states
        else:
            return decoder_out

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding

        x, decoder_padding_mask, pos = self.copying_source_as_inputs(
            prev_output_tokens,
            encoder_out=encoder_out
        )

        return self._extract_features(
            x, decoder_padding_mask, pos,
            encoder_out=encoder_out,
            early_exit=early_exit,
            **unused
        )

    def _extract_features(
            self,
            x, decoder_padding_mask,
            pos=None,
            encoder_out=None,
            early_exit=None,
            tgt_tokens=None,
            **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
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

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def copying_source_as_inputs(self, prev_output_tokens, encoder_out=None, add_position=True, **unused):
        mapping_use = self.map_use
        mapping_func = self.map_func

        if mapping_use.startswith("embed"):
            src_embed = encoder_out.encoder_embedding
        else:
            src_embed = encoder_out.encoder_out.contiguous().transpose(0, 1)

        src_mask = encoder_out.encoder_padding_mask
        src_mask = ~src_mask if src_mask is not None else prev_output_tokens.new_ones(*src_embed.size()[:2]).bool()
        tgt_mask = prev_output_tokens.ne(self.padding_idx)

        if mapping_func == 'uniform':
            states = self.forward_copying_source(
                src_embed, src_mask, prev_output_tokens.ne(self.padding_idx)
            )
        elif mapping_func == "soft":
            length_sources = src_mask.sum(1)
            length_targets = tgt_mask.sum(1)
            mapped_logits = _softcopy_assignment(length_sources, length_targets)  # batch_size, tgt_len, src_len
            states = torch.bmm(mapped_logits, src_embed)
        elif mapping_func == "interpolate":
            mapped_logits = _interpolate_assignment(src_mask, tgt_mask)
            states = torch.bmm(mapped_logits, src_embed)
        else:
            states = None

        x, decoder_padding_mask, positions = self.forward_embedding(prev_output_tokens, states, add_position)
        return x, decoder_padding_mask, positions

    def forward_embedding(self, prev_output_tokens, states=None, add_position=True):
        # embed positions
        positions = None if self.embed_positions is None else self.embed_positions(prev_output_tokens)

        # embed tokens and positions
        x = states
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)

        if positions is not None and add_position:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask, positions

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None, use_ref_len=False, beam_size=1):
        if tgt_tokens is not None and use_ref_len:
            # only used for oracle training.
            length_tgt = tgt_tokens.ne(self.padding_idx).sum(1).long()
            return length_tgt

        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        src_lens = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lens = enc_feats.new_ones(enc_feats.size(1)).fill_(enc_feats.size(0))
            else:
                src_lens = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lens = src_lens.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lens = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lens - src_lens + 128
            else:
                length_tgt = tgt_lens
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            if beam_size == 1:
                pred_lens = length_out.max(-1)[1]
                if self.pred_length_offset:
                    length_tgt = pred_lens - 128 + src_lens
                else:
                    length_tgt = pred_lens
            else:
                pred_lens = length_out.topk(beam_size)[1]  # batch_size, beam_size
                if self.pred_length_offset:
                    src_lens = src_lens.unsqueeze(-1).expand(-1, beam_size)
                    length_tgt = pred_lens - 128 + src_lens
                else:
                    length_tgt = pred_lens
                length_tgt = length_tgt.squeeze(0)

        return length_tgt


@register_model_architecture('awesome_nat', 'awesome_nat_wmt')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.mapping_func = getattr(args, "mapping_func", "uniform")
    args.mapping_use = getattr(args, "mapping_use", "embed")
    from fairseq.models.transformer import base_architecture
    base_architecture(args)


@register_model_architecture('awesome_nat', 'awesome_nat_iwslt')
def awesome_nat_iwslt(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture('awesome_nat', 'awesome_nat_small')
def awesome_nat_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 5)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 256)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    base_architecture(args)


@register_model_architecture('awesome_nat', 'awesome_nat_base')
def awesome_nat_base(args):
    """ original implementation: d_model = d_hidden = 512"""
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('awesome_nat', 'awesome_nat_iwslt16')
def awesome_nat_iwslt16(args):
    """ used in Gu et al. ICLR 2018 """
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 278)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 507)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    args.encoder_layers = getattr(args, 'encoder_layers', 5)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 278)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 507)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 2)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    args.dropout = getattr(args, "dropout", 0.079)
    base_architecture(args)


@register_model_architecture('awesome_nat', 'awesome_nat_iwslt14')
def awesome_nat_iwslt14(args):
    awesome_nat_iwslt(args)


@register_model_architecture('awesome_nat', 'awesome_nat_wmt14')
def awesome_nat_wmt14(args):
    base_architecture(args)
