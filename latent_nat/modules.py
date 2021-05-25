"""
used for Transform with relative position embeddings
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import Namespace
from typing import Dict, Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import (
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
)
from fairseq.modules.multihead_attention import MultiheadAttention
from torch import Tensor
from torch.nn import Parameter

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


class RelativePositionEmbeddings(nn.Module):
    """
    learned relative position embedding for self-attention with relative position of shaw et al
    """

    def __init__(self, max_rel_positions, embedding_dim, dropout=0.0, direction=True, **params):
        super().__init__()
        self.window_size = max_rel_positions
        self.embedding_dim = embedding_dim
        self.direction = direction

        num_embeddings = max_rel_positions * 2 + 1 if self.direction else max_rel_positions + 1
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def map_to_index(self, distance, shift_to_zero=True):
        max_rel_len = self.window_size
        if max_rel_len is None:
            distance = distance
        else:
            distance = distance.clamp(-max_rel_len, max_rel_len)

        if self.direction:
            if shift_to_zero and max_rel_len is not None:
                distance = distance + max_rel_len
            else:
                distance = distance
        else:
            distance = distance.abs()
        return distance

    def forward(self, inputs):
        """
        :param inputs: length, length, num_embeddings or length
        :return:
        """
        if inputs.dim() > 2:
            embed = inputs @ self.embeddings.weight
            embed = self.dropout(embed)
            return embed
        elif inputs.dim() == 2:
            distance = inputs
        else:
            inputs = inputs.squeeze()
            distance = inputs[:, None] - inputs[None, :]

        distance = self.map_to_index(distance)
        embed = self.embeddings(distance)
        embed = self.dropout(embed)
        return embed


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-1)).squeeze(-1)


def shaw_attention(query, key, pos_key):
    """

        :param query:
        :param key:
        :param pos_key: length, length, depth
        :return:
        """
    bsize, heads, length, depth = key.size()

    q_dot_k = matmul(query, key.contiguous().transpose(-1, -2))  # batch, heads, length, length

    query_for_pos = query.contiguous().permute(2, 0, 1, 3).view(length, bsize * heads, depth)
    pos_for_att = pos_key.contiguous().transpose(-2, -1)  # length, depth, length

    q_dot_p = matmul(query_for_pos, pos_for_att)  # length, batch*heads, length
    q_dot_p = q_dot_p.contiguous().permute(1, 0, 2).view(bsize, heads, length, length)

    return q_dot_k + q_dot_p


def shaw_combine(probs, value, pos_val):
    """

    :param probs:
    :param value:
    :param pos_val: length, length, depth
    :return:
    """
    bsize, heads, length, depth = value.size()

    w_dot_v = matmul(probs, value)  # batch, head, length, depth

    w_for_comb = probs.contiguous().permute(2, 0, 1, 3).view(length, bsize * heads, length)
    w_dot_p = matmul(w_for_comb, pos_val)  # length,batch*heads, depth
    w_dot_p = w_dot_p.contiguous().permute(1, 0, 2).view(bsize, heads, length, depth)

    return w_dot_v + w_dot_p


class RelativeSelfAttention(MultiheadAttention):
    """Multi-headed attention with relative attentions.

    See "Self Attention with relative positions" for more details.
    """

    @classmethod
    def relative_attention(cls, query, key, pos_key):
        if pos_key.dim() == 3:
            return shaw_attention(query, key, pos_key)

    @classmethod
    def relative_combine(cls, probs, value, pos_val):
        if pos_val.dim() == 3:
            return shaw_combine(probs, value, pos_val)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            pos_key=None,
            pos_val=None,
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        # self-attention
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k is not None:
            k = (k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))

        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = self.relative_attention(
            q.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            k.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            pos_key,
        ).contiguous().view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf")
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = self.relative_combine(
            probs=attn_probs.contiguous().view(bsz, self.num_heads, tgt_len, src_len),
            value=v.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            pos_val=pos_val
        ).contiguous().view(bsz * self.num_heads, -1, self.head_dim)

        if self.onnx_trace and attn.size(1) == 1:
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None

        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


class FFNAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False):
        super(FFNAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.out = nn.Linear(hidden_dim, 1, bias=bias)
        self._inf = Parameter(torch.Tensor([-1e18]), requires_grad=False)
        self.inf = None

        # Initialize vector V
        nn.init.uniform_(self.out.weight, -1, 1)

    def forward(self, query, key, mask=None):
        query = self.q_proj(query).unsqueeze(2).expand(-1, -1, key.size(1))  # (batch, hidden, seq_len)
        key = key.permute(0, 2, 1)  # (batch, hidden, seq_len)
        key = self.k_proj(key)  # (batch, hidden, seq_len)

        attn_weight = self.out((query + key).permute(0, 2, 1)).squeeze(-1)  # (batch, seq_len)
        if mask is not None and len(attn_weight[mask]) > 0:
            attn_weight[mask] = self.inf[mask]

        attn_prob = attn_weight.softmax(dim=-1)
        attn = torch.bmm(key, attn_prob.unsqueeze(2)).squeeze(2)
        return attn, attn_weight

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class DotProductAttention(nn.Module):
    """ Attention model for Pointer-Net """

    def __init__(self, ninp, nhid):
        """
        Initiate Attention

        :param int ninp: Input's diamention
        :param int nhid: Number of hidden units in the attention
        """

        super(DotProductAttention, self).__init__()

        self.input_dim = ninp
        self.hidden_dim = nhid

        self.input_linear = nn.Linear(ninp, nhid)
        self.context_linear = nn.Conv1d(ninp, nhid, 1, 1)
        self.V = Parameter(torch.FloatTensor(nhid), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([-1e18]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.inf = None

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, inputs, context, mask):
        """
        Attention - Forward-pass

        :param Tensor inputs: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(inputs).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        attn_weight = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if mask is not None and len(attn_weight[mask]) > 0:
            attn_weight[mask] = self.inf[mask]
        attn_prob = self.softmax(attn_weight)

        attn = torch.bmm(ctx, attn_prob.unsqueeze(2)).squeeze(2)

        return attn, attn_weight

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.0):
        super().__init__()
        # dropout = 0.0 # means 17
        self.input_to_hidden = nn.Linear(d_model, d_hidden)
        self.hidden_to_output = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        h = F.relu(self.input_to_hidden(inputs))
        h = self.dropout(h)
        return self.hidden_to_output(h)


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


class TransformerDecoderFunc(nn.Module):
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
            learned=args.decoder_learned_pos
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
            for _ in range(args.decoder_layers)
        ])
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(args, "no_decoder_final_norm", False):
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

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
        inputs = self.preprocess(inputs=inputs, decoder_padding_mask=decoder_padding_mask)
        return self._extract_features(
            inputs, decoder_padding_mask, encoder_out
        )

    def preprocess(self, inputs, decoder_padding_mask, include_pos=False):
        if include_pos or self.args.self_attn_cls == "shaw":
            return inputs
        mask = decoder_padding_mask.long()
        pos_tokens = (mask * self.padding_idx) + (1 - mask) * (self.padding_idx + 1)
        inputs = self.forward_embedding(inputs, pos_tokens)[0]
        return inputs

    def _extract_features(
            self,
            inputs,
            decoder_padding_mask,
            encoder_out=None,
    ):
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
