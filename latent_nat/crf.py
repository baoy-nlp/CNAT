"""Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
"""

import torch
import torch.nn as nn
from fairseq.modules import DynamicCRF
from fairseq.modules.dynamic_crf_layer import logsumexp


class CRF(nn.Module):
    def __init__(self, num_embedding: int, beam_size=64, batch_first=True, **unused):
        super().__init__()
        self.num_tags = num_embedding
        self.batch_first = batch_first
        self.beam = beam_size

        self.start_transitions = nn.Parameter(torch.empty(num_embedding))
        self.end_transitions = nn.Parameter(torch.empty(num_embedding))
        self.transitions = nn.Parameter(torch.empty(num_embedding, num_embedding))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def extra_repr(self):
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions, targets, masks=None, beam=None) -> torch.Tensor:
        """
        Compute the conditional log-likelihood of a sequence of target tokens given emission scores

        Args:
            emissions (`~torch.Tensor`): Emission score are usually the unnormalized decoder output
                ``(batch_size, seq_len, vocab_size)``. We assume batch-first
            targets (`~torch.LongTensor`): Sequence of target token indices
                ``(batch_size, seq_len)
            masks (`~torch.ByteTensor`): Mask tensor with the same size as targets

        Returns:
            `~torch.Tensor`: approximated log-likelihood
        """
        emissions, masks, targets = self._validate(emissions, masks=masks, targets=targets)
        numerator = self._compute_score(emissions, targets, masks)
        denominator = self._compute_normalize(emissions, masks)
        return numerator - denominator

    def forward_decoder(self, emissions: torch.Tensor, masks=None):
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            masks (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        emissions, masks, _ = self._validate(emissions, masks=masks)
        return self._viterbi_decode(emissions, masks)

    def extract_feature(self, emissions, targets, masks=None, beam=None, log_prob=True):
        _emissions, _masks, _targets = self._validate(emissions, masks=masks, targets=targets)
        numerator = self._compute_score(_emissions, _targets, _masks)
        f_sum, forward = self._compute_forward(_emissions, _masks)
        crf_nll = -(numerator - f_sum)
        crf_nll = (crf_nll / masks.type_as(crf_nll).sum(-1)).mean()
        if not log_prob and self.training:
            return {
                "CRF": {
                    "loss": crf_nll,
                    # "out": log_prob,
                    "tgt": targets,
                    "mask": masks,
                    "factor": 1.0,
                },
            }

        b_sum, backward = self._compute_backward(_emissions, _masks)
        forward_tensor = torch.stack(forward, dim=1)
        backward_tensor = torch.stack(backward, dim=1)  # batch_size, seq_len, _

        # log_prob = (forward_tensor + backward_tensor) - f_sum[:, None, None]
        z = forward_tensor + backward_tensor
        log_prob = z - logsumexp(z, dim=-1)[:, :, None]
        return {
            "CRF": {
                "loss": crf_nll,
                "out": log_prob,
                "tgt": targets,
                "mask": masks,
                "factor": 1.0,
            },
            "log_prob": log_prob,
        }

    def _compute_score(self, emissions, targets, masks):
        seq_length, batch_size = targets.shape
        masks = masks.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[targets[0]]
        score += emissions[0, torch.arange(batch_size), targets[0]]

        for i in range(1, seq_length):
            score += self.transitions[targets[i - 1], targets[i]] * masks[i]
            score += emissions[i, torch.arange(batch_size), targets[i]] * masks[i]

        seq_ends = masks.long().sum(dim=0) - 1
        last_tags = targets[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalize(self, emissions, masks=None):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and masks.dim() == 2
        assert emissions.shape[:2] == masks.shape
        assert emissions.size(2) == self.num_tags

        seq_len = emissions.size(0)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = logsumexp(next_score, dim=1)

            if masks is not None:
                score = torch.where(masks[i].unsqueeze(1), next_score, score)
            else:
                score = next_score

        score += self.end_transitions
        return logsumexp(score, dim=1)

    def _compute_forward(self, emissions, masks=None):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_len = emissions.size(0)
        forward = []
        score = self.start_transitions + emissions[0]
        # score = emissions[0]

        for i in range(1, seq_len):
            forward.append(score)
            step_score = self.transitions + emissions[i].unsqueeze(1)
            next_score = score.unsqueeze(2) + step_score  # batch_size, K, K
            next_score = logsumexp(next_score, dim=1)  # batch_size, k

            if masks is not None:
                score = torch.where(masks[i].unsqueeze(1), next_score, score)
            else:
                score = next_score

        forward.append(score)
        score += self.end_transitions
        score = logsumexp(score, dim=1)
        return score, forward

    def _compute_backward(self, emissions, masks=None):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_len, batch_size, _ = emissions.size()
        backward = []
        score = self.end_transitions.unsqueeze(0).expand(batch_size, -1)  # batch_size, K
        # score = emissions.new_zeros(batch_size, emissions.size(-1))

        for i in range(seq_len - 2, -1, -1):
            backward.append(score)
            step_score = self.transitions + emissions[i + 1].unsqueeze(1)  # batch_size, K, K
            next_score = score.unsqueeze(1) + step_score
            next_score = logsumexp(next_score, dim=2)

            if masks is not None:
                score = torch.where(masks[i].unsqueeze(1), next_score, score)
            else:
                score = next_score

        backward.append(score)
        backward.reverse()

        score += (self.start_transitions + emissions[0])
        # score += emissions[0]
        score = logsumexp(score, dim=1)

        return score, backward

    def _viterbi_decode(self, emissions, masks, beam=None):
        assert emissions.dim() == 3 and masks.dim() == 2
        assert emissions.shape[:2] == masks.shape
        assert emissions.size(2) == self.num_tags

        seq_length, batch_size = masks.shape

        score = self.start_transitions + emissions[0]
        traj_tokens, traj_scores = [], []
        finalized_tokens, finalized_scores = [], []

        for i in range(1, seq_length):
            traj_scores.append(score)
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            if masks is not None:
                score = torch.where(masks[i].unsqueeze(1), next_score, score)
            else:
                score = next_score
            traj_tokens.append(indices)

        score += self.end_transitions

        best_score, best_index = score.max(dim=1)
        finalized_tokens.append(best_index[:, None])
        finalized_scores.append(best_score[:, None])

        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
            previous_index = finalized_tokens[-1]
            finalized_tokens.append(idx.gather(1, previous_index))
            finalized_scores.append(scs.gather(1, previous_index))

        finalized_tokens.reverse()
        finalized_tokens = torch.cat(finalized_tokens, 1)

        finalized_scores.reverse()
        finalized_scores = torch.cat(finalized_scores, 1)
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]

        return finalized_scores, finalized_tokens

    def _validate(self, emissions, targets=None, masks=None):
        if masks is None:
            masks = torch.ones_like(emissions.shape[:2], dtype=torch.uint8).to(emissions)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            masks = masks.transpose(0, 1)
            if targets is not None:
                targets = targets.transpose(0, 1)

        return emissions, masks, targets


class WrapperCRF(DynamicCRF):
    def __init__(self, num_embedding, low_rank=32, beam_size=64, **unused):
        super().__init__(num_embedding, low_rank, beam_size)


class BiaffineCRF(nn.Module):
    """
    NUM_TAGS * NUM_HIDDEN
    XU * (XV)^T
    """

    def __init__(self, num_embedding: int, num_head: int = 8, embed_size=512, batch_first=True, **unused):
        super().__init__()
        self.num_cluster = num_embedding
        self.batch_first = batch_first

        self.num_channel = num_head
        self.num_hidden = embed_size // num_head
        self.E1 = nn.Linear(self.num_hidden, self.num_cluster)
        self.E2 = nn.Linear(self.num_hidden, self.num_cluster)

        self.start_transitions = nn.Parameter(torch.empty(self.num_cluster))
        self.end_transitions = nn.Parameter(torch.empty(self.num_cluster))
        self.transition = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    @property
    def transitions(self):
        assert self.transition is not None, "need compute transition at first"
        return self.transition

    def set_transitions(self, inputs=None):
        L, B, D = inputs.size()
        flatten = inputs.contiguous().view(L * B, self.num_channel, -1)
        e1 = self.E1(flatten)  # L * B , num_head, num_tags
        e2 = self.E2(flatten)  # L * B , num_head, num_tags

        res = e1.transpose(1, 2) @ e2  # L*B, num_tags, num_tags
        self.transition = res.contiguous().view(L, B, self.num_cluster, self.num_cluster)

    def extra_repr(self):
        return f'{self.__class__.__name__}(num_tags={self.num_cluster})'

    def forward(self, inputs, emissions, targets, masks=None, beam=None) -> torch.Tensor:
        # use for inference
        inputs, emissions, masks, targets = self._validate(inputs, emissions, masks=masks, targets=targets)
        self.set_transitions(inputs)
        numerator = self._compute_score(emissions, targets, masks)
        denominator = self._compute_normalize(emissions, masks)
        return numerator - denominator

    def forward_decoder(self, inputs, emissions, masks=None):
        # use for inference
        inputs, emissions, masks, _ = self._validate(inputs, emissions, masks=masks)
        self.set_transitions(inputs)
        return self._viterbi_decode(emissions, masks)

    def extract_feature(self, inputs, emissions, targets, masks=None, beam=None, log_prob=True):
        # independent with forward algorithm
        _inputs, _emissions, _masks, _targets = self._validate(inputs, emissions, masks=masks, targets=targets)
        self.set_transitions(_inputs)
        numerator = self._compute_score(_emissions, _targets, _masks)
        f_sum, forward = self._compute_forward(_emissions, _masks)
        crf_nll = -(numerator - f_sum)
        crf_nll = (crf_nll / masks.type_as(crf_nll).sum(-1)).mean()
        if not log_prob and self.training:
            return {
                "CRF": {
                    "loss": crf_nll,
                    "tgt": targets,
                    "mask": masks,
                    "factor": 1.0,
                },
            }

        b_sum, backward = self._compute_backward(_emissions, _masks)
        forward_tensor = torch.stack(forward, dim=1)
        backward_tensor = torch.stack(backward, dim=1)  # batch_size, seq_len, _

        # log_prob = (forward_tensor + backward_tensor) - f_sum[:, None, None]
        z = forward_tensor + backward_tensor
        log_prob = z - logsumexp(z, dim=-1)[:, :, None]
        return {
            "CRF": {
                "loss": crf_nll,
                "out": log_prob,
                "tgt": targets,
                "mask": masks,
                "factor": 1.0,
            },
            "log_prob": log_prob,
        }

    def _compute_transitions(self, inputs, masks=None):
        """

        :param inputs: [sequence_length, batch_size, hidden_dim]
        :param masks:
        :return:
            sequence_length, batch_size, num_tags, num_tags
        """
        pass

    def _compute_score(self, emissions, targets, masks):
        seq_length, batch_size = targets.shape
        masks = masks.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[targets[0]]
        score += emissions[0, torch.arange(batch_size), targets[0]]

        for i in range(1, seq_length):
            transition = self.transitions[i]  # batch_size, num_tags, num_tags
            score += transition[range(batch_size), targets[i - 1], targets[i]] * masks[i]
            score += emissions[i, torch.arange(batch_size), targets[i]] * masks[i]

        seq_ends = masks.long().sum(dim=0) - 1
        last_tags = targets[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalize(self, emissions, masks=None):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and masks.dim() == 2
        assert emissions.shape[:2] == masks.shape
        assert emissions.size(2) == self.num_cluster

        seq_len = emissions.size(0)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_len):
            transition = self.transitions[i]  # batch_size, num_tags,num_tags
            broadcast_score = score.unsqueeze(2)  # batch_size, num_tags, 1
            broadcast_emissions = emissions[i].unsqueeze(1)  # batch_size, 1, num_tags
            next_score = broadcast_score + transition + broadcast_emissions
            next_score = logsumexp(next_score, dim=1)

            if masks is not None:
                score = torch.where(masks[i].unsqueeze(1), next_score, score)
            else:
                score = next_score

        score += self.end_transitions
        return logsumexp(score, dim=1)

    def _compute_forward(self, emissions, masks=None):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_len = emissions.size(0)
        forward = []
        score = self.start_transitions + emissions[0]
        # score = emissions[0]

        for i in range(1, seq_len):
            forward.append(score)
            transition = self.transitions[i]
            step_score = transition + emissions[i].unsqueeze(1)
            next_score = score.unsqueeze(2) + step_score  # batch_size, K, K
            next_score = logsumexp(next_score, dim=1)  # batch_size, k

            if masks is not None:
                score = torch.where(masks[i].unsqueeze(1), next_score, score)
            else:
                score = next_score

        forward.append(score)
        score += self.end_transitions
        score = logsumexp(score, dim=1)
        return score, forward

    def _compute_backward(self, emissions, masks=None):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_len, batch_size, _ = emissions.size()
        backward = []
        score = self.end_transitions.unsqueeze(0).expand(batch_size, -1)  # batch_size, K
        # score = emissions.new_zeros(batch_size, emissions.size(-1))

        for i in range(seq_len - 2, -1, -1):
            backward.append(score)
            transition = self.transitions[i + 1]
            step_score = transition + emissions[i + 1].unsqueeze(1)  # batch_size, K, K
            next_score = score.unsqueeze(1) + step_score
            next_score = logsumexp(next_score, dim=2)

            if masks is not None:
                score = torch.where(masks[i].unsqueeze(1), next_score, score)
            else:
                score = next_score

        backward.append(score)
        backward.reverse()

        score += (self.start_transitions + emissions[0])
        # score += emissions[0]
        score = logsumexp(score, dim=1)

        return score, backward

    def _viterbi_decode(self, emissions, masks, beam=None):
        assert emissions.dim() == 3 and masks.dim() == 2
        assert emissions.shape[:2] == masks.shape
        assert emissions.size(2) == self.num_cluster

        seq_length, batch_size = masks.shape

        score = self.start_transitions + emissions[0]
        traj_tokens, traj_scores = [], []
        finalized_tokens, finalized_scores = [], []

        for i in range(1, seq_length):
            traj_scores.append(score)
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            transition = self.transitions[i]
            next_score = broadcast_score + transition + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            if masks is not None:
                score = torch.where(masks[i].unsqueeze(1), next_score, score)
            else:
                score = next_score
            traj_tokens.append(indices)

        score += self.end_transitions

        best_score, best_index = score.max(dim=1)
        finalized_tokens.append(best_index[:, None])
        finalized_scores.append(best_score[:, None])

        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
            previous_index = finalized_tokens[-1]
            finalized_tokens.append(idx.gather(1, previous_index))
            finalized_scores.append(scs.gather(1, previous_index))

        finalized_tokens.reverse()
        finalized_tokens = torch.cat(finalized_tokens, 1)

        finalized_scores.reverse()
        finalized_scores = torch.cat(finalized_scores, 1)
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]

        return finalized_scores, finalized_tokens

    def _validate(self, inputs, emissions, targets=None, masks=None):
        if masks is None:
            masks = torch.ones_like(emissions.shape[:2], dtype=torch.uint8).to(emissions)

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
            emissions = emissions.transpose(0, 1)
            masks = masks.transpose(0, 1)
            if targets is not None:
                targets = targets.transpose(0, 1)

        return inputs, emissions, masks, targets


def crf_training(crf_layer, inputs, emission, tgt_tokens, forward_mask, log_prob=True, include_emission_loss=False):
    if hasattr(crf_layer, "extract_feature"):
        extract_feature = crf_layer.extract_feature
        if isinstance(crf_layer, BiaffineCRF):
            # include an extra encoding process for inputs
            ret = extract_feature(
                inputs=inputs, emissions=emission, targets=tgt_tokens, masks=forward_mask, log_prob=log_prob
            )
        else:
            ret = extract_feature(
                emissions=emission, targets=tgt_tokens, masks=forward_mask, log_prob=log_prob
            )
    else:
        ret = {}
        crf_nll = -crf_layer(emissions=emission, targets=tgt_tokens, masks=forward_mask)
        crf_nll = (crf_nll / forward_mask.type_as(crf_nll).sum(-1)).mean()
        ret['CRF'] = {
            'loss': crf_nll,
            'factor': 1.0
        }
    ret['out'] = emission  # used for schedule sampling
    if include_emission_loss:
        ret["CRF-emission"] = {
            "out": emission,
            "tgt": tgt_tokens,
            "mask": forward_mask,
            "factor": 0.5
        }
    elif 'log_prob' in ret:
        ret['out'] = ret['log_prob']
    return ret


def crf_inference(crf_layer, inputs, emission, forward_mask):
    if isinstance(crf_layer, BiaffineCRF):
        return crf_layer.forward_decoder(
            inputs=inputs,
            emissions=emission,
            masks=forward_mask
        )
    else:
        return crf_layer.forward_decoder(
            emissions=emission,
            masks=forward_mask
        )


def build_crf_layer(args):
    return CRF_CLS[getattr(args, "crf_cls", "DCRF")](
        num_embedding=args.num_cdes,
        low_rank=getattr(args, "crf_lowrank_approx", 32),
        beam_size=getattr(args, "crf_beam_approx", 64),
        num_head=getattr(args, "crf_num_head", 8),
        embed_size=args.decoder_embed_dim
    )


CRF_CLS = {
    "DCRF": WrapperCRF,
    "BCRF": BiaffineCRF
}
