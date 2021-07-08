import math

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion


@register_criterion("awesome_nat_loss")
class GenericNATCriterion(LabelSmoothedDualImitationCriterion):
    """
        include the accuracy if needed
    """

    @classmethod
    def _compute_acc(cls, tgt, out=None, mask=None, pred=None, name="acc"):
        if out is None and pred is None:
            correct, totals = tgt.new_tensor(0), tgt.new_tensor(0)
        else:
            tgt = tgt.view(-1)
            if pred is None and out is not None:
                out = out.view(-1, out.size(-1))
                pred = out.max(-1)[1]

            indicate = pred.eq(tgt)

            if mask is not None:
                mask = mask.view(-1)
                indicate = indicate * mask
                totals = mask.sum()
            else:
                totals = tgt.new_tensor(tgt.size(0))
            correct = indicate.sum()
        if totals.item() > 0:
            acc = correct / (totals + 1e-3)
        else:
            acc = tgt.new_tensor(0)
        return {"name": name, "correct": correct, "count": totals, "acc": acc}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss = [], []
        accuracies = []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )

            if not model.training and "tgt" in outputs[obj]:
                if not outputs[obj].get("no-acc", False):
                    acc = self._compute_acc(
                        pred=outputs[obj].get("pred", None),
                        out=outputs[obj].get("out", None),
                        tgt=outputs[obj].get("tgt", None),
                        mask=outputs[obj].get("mask", None),
                        name=obj + '-acc'
                    )
                    accuracies += [acc]
            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(_loss["loss"] for _loss in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
            else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )
        for l in accuracies:
            logging_output["{}-correct".format(l["name"])] = l["correct"]
            logging_output["{}-count".format(l["name"])] = l["count"]

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar("loss", loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )
            if key.endswith("-count"):
                # support MULIT-GPU
                name = key[:-6]
                count = utils.item(sum(log.get("{}-count".format(name), 0) for log in logging_outputs))
                correct = utils.item(sum(log.get("{}-correct".format(name), 0) for log in logging_outputs))
                val = correct * 100.0 / count if count > 0 else 0.0
                metrics.log_scalar(name, val, sample_size, round=2)
