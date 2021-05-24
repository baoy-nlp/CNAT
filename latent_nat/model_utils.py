from argparse import Namespace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.lstm import (
    AdaptiveSoftmax,
    AttentionLayer,
    Embedding,
    Linear,
    LSTMCell,
    LSTM,
    LSTMDecoder
)
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
)
from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


class VanillaTransformerEncoder(TransformerEncoder):
    def __init__(
            self, dictionary, embed_tokens,
            encoder_layers, encoder_embed_dim, encoder_ffn_embed_dim,
            dropout, encoder_layerdrop,
            max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
            no_scale_embedding=False, no_token_positional_embeddings=False, layernorm_embedding=False,
            adaptive_input=False, quant_noise_pq=0, quant_noise_pq_block_size=8,
            encoder_normalize_before=False,
            activation_fn="relu", activation_dropout=0, relu_dropout=0,
            mask_self=False,
            **unused
    ):
        args = Namespace()
        args.encoder_layers = encoder_layers
        args.encoder_embed_dim = encoder_embed_dim
        args.encoder_ffn_embed_dim = encoder_ffn_embed_dim,
        args.dropout = dropout
        args.encoder_layerdrop = encoder_layerdrop
        args.max_source_positions = max_source_positions
        args.no_scale_embedding = no_scale_embedding
        args.no_token_positional_embeddings = no_token_positional_embeddings
        args.layernorm_embedding = layernorm_embedding
        args.adaptive_input = adaptive_input
        args.quant_noise_pq = quant_noise_pq
        args.quant_noise_pq_block_size = quant_noise_pq_block_size,
        args.encoder_normalize_before = encoder_normalize_before
        args.activation_fn = activation_fn
        args.activation_dropout = activation_dropout
        args.relu_dropout = relu_dropout

        super().__init__(args, dictionary, embed_tokens)
        self.mask_self = mask_self

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, x=None, encoder_padding_mask=None):
        if x is None:
            x, encoder_embedding = self.forward_embedding(src_tokens)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
        else:
            encoder_embedding = None

        # compute padding mask
        if encoder_padding_mask is None:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask,
                attn_mask=self.buffered_self_mask(x)
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    def buffered_self_mask(self, tensor):
        if self.mask_self:
            return None
        dim = tensor.size(0)
        mask = utils.fill_with_neg_inf(torch.zeros([dim, dim])).to(tensor)
        diag_ids = torch.eye(dim).to(tensor) == 1
        return mask[diag_ids]


def compute_margin(dist1, dist2, mask, tgt, log_diff=False, use_info_exp=False, info_bound=-1.0, is_prob=False):
    device = tgt.device
    mask = mask.to(device)
    tgt = tgt.to(device)[mask]

    if dist1 is None:
        log_dist2 = F.softmax(dist2[mask], dim=-1) if not is_prob else dist2[mask]
        diff = (1.0 - log_dist2).view(-1, log_dist2.size(-1))  # 1 - prob
    else:
        if log_diff:
            log_dist1 = F.log_softmax(dist1[mask], dim=-1)
            log_dist2 = F.log_softmax(dist2[mask], dim=-1)
        else:
            log_dist1 = F.softmax(dist1[mask], dim=-1)
            log_dist2 = F.softmax(dist2[mask], dim=-1)
        diff = (log_dist1 - log_dist2).view(-1, log_dist2.size(-1))
    margin = diff[range(diff.size(0)), tgt.view(-1)].view(*tgt.size())
    if info_bound < 0 and not use_info_exp:
        return margin
    loss = margin.mean()
    if use_info_exp:
        return loss.exp()

    if info_bound > 0:
        loss = (loss + info_bound).abs()
    return loss


def weighted_add(add_item1, add_item2, weight=None):
    if weight is None:
        return add_item1 + add_item2
    else:
        return add_item1 * (1 - weight) + weight * add_item2


class VanillaTransformerDecoder(nn.Module):
    def __init__(self, args, padding_idx, embed_positions=None):
        super().__init__()
        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.embed_dim = args.decoder_embed_dim
        self.padding_idx = padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_positions = PositionalEmbedding(
            self.max_target_positions,
            self.embed_dim,
            self.padding_idx,
            learned=args.predictor_learned_pos
        ) if embed_positions is None else embed_positions

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = False

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn=False)
            for _ in range(args.predictor_layers)
        ])
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(args, "no_decoder_final_norm", False):
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

        self.output_projection = nn.Linear(self.embed_dim, args.vq_num, bias=False)

    def forward_embedding(self, inputs, pos_tokens):
        if self.embed_positions is not None:
            if inputs is not None:
                inputs = inputs + self.embed_positions(pos_tokens)
            else:
                inputs = self.embed_positions(pos_tokens)

        if self.layernorm_embedding is not None:
            inputs = self.layernorm_embedding(inputs)
        inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        return inputs, None

    def forward(
            self,
            inputs,
            decoder_padding_mask,
            encoder_out=None,
    ):
        mask = decoder_padding_mask.long()
        pos_tokens = (mask * self.padding_idx) + (1 - mask) * (self.padding_idx + 1)

        inputs, _ = self.forward_embedding(inputs, pos_tokens)

        inputs = inputs.transpose(0, 1)

        attn, inner_states = None, []

        alignment_layer = self.num_layers - 1
        for idx, layer in enumerate(self.layers):

            self_attn_mask = None

            inputs, layer_attn, _ = layer(
                inputs,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state=None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=decoder_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(inputs)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(inputs)

        if attn is not None:
            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            inputs = self.layer_norm(inputs)

        inputs = inputs.transpose(0, 1)
        return inputs, {"attn": [attn], "inner_states": inner_states}


class VanillaLSTMEncoder(nn.Module):
    def __init__(
            self, embed_dim=512, hidden_size=512, num_layers=1,
            dropout_in=0.1, dropout_out=0.1, bidirectional=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_in_module = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out_module.p if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.padding_idx = -1
        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, x: Tensor, padding_mask: Tensor, enforce_sorted: bool = True):
        bsz, seqlen, _ = x.size()
        x = self.dropout_in_module(x)
        src_lengths = (~padding_mask).long().sum(dim=-1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.data, enforce_sorted=enforce_sorted
        )

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx * 1.0)
        x = self.dropout_out_module(x)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, bsz)
            final_cells = self.combine_bidir(final_cells, bsz)

        encoder_padding_mask = padding_mask.t()

        return tuple((
            x,  # seq_len x batch x hidden
            final_hiddens,  # num_layers x batch x num_directions*hidden
            final_cells,  # num_layers x batch x num_directions*hidden
            encoder_padding_mask,  # seq_len x batch
        ))

    def combine_bidir(self, outs, bsz: int):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple((
            encoder_out[0].index_select(1, new_order),
            encoder_out[1].index_select(1, new_order),
            encoder_out[2].index_select(1, new_order),
            encoder_out[3].index_select(1, new_order),
        ))


class VanillaLSTMDecoder(LSTMDecoder):
    def __init__(
            self, num_embeddings, padding_idx, embed_dim=512, hidden_size=512, out_embed_dim=512,
            num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
            encoder_output_units=512, pretrained_embed=None,
            share_input_output_embed=False, adaptive_softmax_cutoff=None,
            max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
            residuals=False,
    ):
        super(LSTMDecoder, self).__init__(dictionary=None)
        self.dropout_in_module = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = attention
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers

        self.adaptive_softmax = None
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=input_feed_size + embed_dim if layer == 0 else hidden_size, hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        if attention:
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None

        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings, hidden_size, adaptive_softmax_cutoff, dropout=dropout_out,
            )
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(
            self,
            prev_output_tokens,
            encoder_out: EncoderOut = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            src_lengths: Optional[Tensor] = None,
            input_feed: Optional[Tensor] = None
    ):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: EncoderOut = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.
        """
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out.encoder_out
            encoder_padding_mask = encoder_out.encoder_padding_mask
        else:
            encoder_outs = torch.empty(0)
            encoder_padding_mask = torch.empty(0)
        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        # elif encoder_out is not None:
        #     # setup recurrent cells
        #     prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
        #     prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
        #     if self.encoder_hidden_proj is not None:
        #         prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]
        #         prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
        #     input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = None

        assert srclen > 0 or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            }
        )
        self.set_incremental_state(incremental_state, 'cached_state', cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores


class InputableLSTMDecoder(nn.Module):
    def __init__(
            self, num_embeddings, padding_idx, embed_dim=512, hidden_size=512, out_embed_dim=512,
            input_feed_size=512, num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
            encoder_output_units=512, pretrained_embed=None,
            share_input_output_embed=False, adaptive_softmax_cutoff=None,
            max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
            residuals=False,
    ):
        super().__init__()
        self.dropout_in_module = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = attention
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers
        self.adaptive_softmax = None
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed
        self.input_feed_size = input_feed_size
        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=input_feed_size + embed_dim if layer == 0 else hidden_size, hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        if attention:
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None

        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings, hidden_size, adaptive_softmax_cutoff, dropout=dropout_out,
            )
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward_step(self, x, prev_hiddens, prev_cells, encoder_outs, encoder_padding_mask, input_feed=None):
        x = self.embed_tokens(x)
        x = self.dropout_in_module(x)
        if input_feed is not None:
            feed = torch.cat((x, input_feed), dim=1)
        else:
            feed = x

        for i, rnn in enumerate(self.layers):
            # recurrent cell
            hidden, cell = rnn(feed, (prev_hiddens[i], prev_cells[i]))

            # hidden state becomes the input to the next layer
            feed = self.dropout_out_module(hidden)
            if self.residuals:
                feed = feed + prev_hiddens[i]

            # save state for next time step
            prev_hiddens[i] = hidden
            prev_cells[i] = cell

            # apply attention using the last layer's hidden state
        if self.attention is not None:
            out, attn_score = self.attention(prev_hiddens[len(self.layers) - 1], encoder_outs, encoder_padding_mask)
        else:
            attn_score = None
            out = prev_hiddens
        out = self.dropout_out_module(out)
        return out, attn_score

    def forward(self, prev_output_tokens, encoder_out, inputs=None):
        if encoder_out is not None:
            encoder_outs = encoder_out.encoder_out  # src_len, batch_size, hidden_dim
            encoder_padding_mask = encoder_out.encoder_padding_mask.transpose(0, 1)
        else:
            encoder_outs = torch.empty(0)
            encoder_padding_mask = torch.empty(0)
        bsz, seq_len, _ = inputs.size()

        if encoder_out is None:
            # setup recurrent cells
            zero_state = inputs.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
        else:
            # zero initial
            # setup zero cells, since there is no encoder
            zero_state = inputs.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]

        outs, logits, attn_scores = [], [], []

        for j in range(seq_len):
            out, attn_score = self.forward_step(
                x=prev_output_tokens[:, j],
                prev_hiddens=prev_hiddens,
                prev_cells=prev_cells,
                encoder_outs=encoder_outs,
                encoder_padding_mask=encoder_padding_mask,
                input_feed=inputs[:, j, :] if self.input_feed_size > 0 else None
            )
            outs.append(out)
            attn_scores.append(attn_score)
            if prev_output_tokens.size(1) != seq_len:
                # means at the inference stage
                logit = self.output_layer(out)  # batch_size, vocab_size
                x_j = logit.max(dim=-1)[1].view(-1, 1)
                logits.append(logit)
                prev_output_tokens = torch.cat([prev_output_tokens, x_j], dim=-1)

        if not self.training and self.need_attn and self.attention is not None:
            assert attn_scores is not None
            attn_scores = torch.stack(attn_scores, dim=0)
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        if len(logits) == seq_len - 1:
            logit = self.output_layer(outs[-1])  # batch_size, vocab_size
            logits.append(logit)

        if len(logits) == seq_len:
            logits = torch.stack(logits, dim=1)  # B * T * C
        else:
            outs = torch.stack(outs, dim=1)  # B * T * H
            logits = self.output_layer(outs)
        return logits, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x
