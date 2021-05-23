import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

try:
    from fairseq.models.transformer import EncoderOut
except ImportError:
    from fairseq.models.fairseq_encoder import EncoderOut


class GlobalNames(object):
    MEAN = "mean"
    LOGV = "logv"
    Z = "z"
    REC = "rec"
    PRI_RET = "prior_ret"
    POST_RET = "post_ret"
    INPUT = "inputs"  # inputs of decoder
    FEATURES = "features"  # output of decoder
    PREDICTS = "predicts"  # prediction of decoder
    GLANCING_INPUTS = "glancing_inputs"
    GLANCING_MASK = "glancing_mask"


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

    def forward(self, targets, mask, ratio=0.5, logits=None):
        return glancing_sampling(
            targets=targets, padding_mask=mask, ratio=ratio, logits=logits, n_mode=self.num_mode,
            s_mode=self.sub_mode
        )

    def substitution(self, inputs, ref, observed, pred=None, s_mode=None):
        s_mode = self.sub_mode if s_mode is None else s_mode

        if s_mode == "schedule":
            assert pred is not None, "schedule needs prediction"
            inputs = pred
        return (1 - observed) * inputs + observed * ref


def glancing_sampling(targets, padding_mask, ratio=0.5, logits=None, n_mode="adaptive", s_mode="uniform"):
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

    if s_mode == "uniform":
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
        sample = None

    return sample


def reparameterize(mean, var, is_logv=False, sample_size=1):
    if sample_size > 1:
        mean = mean.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, mean.size(-1))
        var = var.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, var.size(-1))

    if not is_logv:
        sigma = torch.sqrt(var + 1e-10)
    else:
        sigma = torch.exp(0.5 * var)

    epsilon = torch.randn_like(sigma)
    z = mean + epsilon * sigma
    return z


class GaussianVariable(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Linear(input_dim, latent_dim)
        self.logv = nn.Linear(input_dim, latent_dim)
        self.rec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, output_dim),
            nn.Tanh(),
        )

    def forward(self, inputs, max_posterior=False, **kwargs):
        """
        :param inputs:  batch_size,input_dim
        :param max_posterior:
        :return:
            mean: batch_size, latent_dim
            logv: batch_size, latent_dim
            z: batch_size, latent_dim
            rec: batch_size, output_dim
        """
        mean, logv, z = self.posterior(inputs, max_posterior=max_posterior)

        rec = self.rec(z)

        return {GlobalNames.MEAN: mean, GlobalNames.LOGV: logv, GlobalNames.Z: z, GlobalNames.REC: rec}

    def posterior(self, inputs, max_posterior=False):
        mean = self.mean(inputs)
        logv = self.logv(inputs)
        z = reparameterize(mean, logv, is_logv=True) if not max_posterior else mean
        return mean, logv, z

    def prior(self, inputs, n=-1):
        if n < 0:
            n = inputs.size(0)
        z = torch.randn([n, self.latent_dim])

        if inputs is not None:
            z = z.to(inputs)

        return z


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


class SelfATTEncoder(nn.Module):
    """
    remove embedding layer
    The args need includes:
        - dropout
        - encoder_layer_drop
        - encoder_layers

        - embed_dim or encoder_embed_dim
        - encoder_attention_heads
        - attention_dropout
        - encoder_normalize_before
        - encoder_ffn_embed_dim

    """

    def __init__(self, args):
        super().__init__()
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.encoder_layerdrop = args.encoder_layerdrop

        self.embed_dim = getattr(args, "embed_dim", args.encoder_embed_dim)

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer(args) for _ in range(args.latent_layers)])
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

    def forward(self, embedding, padding_mask, return_all_hiddens: bool = False):

        x = embedding.transpose(0, 1)
        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=padding_mask,  # B x T
            encoder_embedding=embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


class MultiATTDecoder(nn.Module):
    """
    """

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
