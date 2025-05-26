import torch


from grammar_env.criterion.criterion import Criterion
from env import Environment


__all__ = ['F1Criterion']


def f1_score(
        gold_spans: set[tuple[int, int]], pred_spans: set[tuple[int, int]], whole_span: tuple[int, int]
) -> float:
    # Ignore the whole span (span covering the whole sentence)
    gold_spans = gold_spans - {whole_span}
    pred_spans = pred_spans - {whole_span}

    tp: int = len(gold_spans & pred_spans)
    if tp == 0:
        return 0.
    precision: float = tp / len(pred_spans)
    recall: float = tp / len(gold_spans)
    return 2 * precision * recall / (precision + recall)


class F1Criterion(Criterion):
    def __init__(
            self, device: torch.device,
    ):
        super().__init__(
            device
        )

    def score_sentences(
            self, env : Environment
    ) -> torch.Tensor:
        pred_spans: list[list[tuple[int, int]]] = env.spans_lists
        gold_spans: list[set[tuple[int, int]]] = [set(sspans.keys()) for sspans in env.gt_spans]

        f1_scores: list[float] = []
        for gt_spans, s_pred_spans, s_len in zip(gold_spans, pred_spans, env.sentence_lengths):
            
            pred_spans_set: set[tuple[int, int]] = set(s_pred_spans)
            whole_span: tuple[int, int] = (0, s_len - 1)
            f1_scores.append(f1_score(gt_spans, pred_spans_set, whole_span))
        f1_scores = torch.tensor(f1_scores, device=self.device)
        self.update_score(f1_scores)
        return f1_scores
