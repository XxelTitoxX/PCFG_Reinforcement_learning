import torch

from grammar_env.corpus.corpus import Corpus
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
            self, corpus: Corpus, device: torch.device,
    ):
        super().__init__(
            corpus, device
        )

    def score_sentences(
            self, env : Environment
    ) -> torch.Tensor:
        pred_spans: list[list[tuple[int, int]]] = env.spans_lists

        f1_scores: list[float] = []
        for s_idx, s_pred_spans, s_len in zip(env.sentence_idx, pred_spans, env.sentence_lengths):
            gold_spans: set[tuple[int, int]] = set(
                (gold_span.start, gold_span.end)
                for gold_span in self.corpus.sentences[s_idx].gold_spans
            )
            s_pred_spans_set: set[tuple[int, int]] = set(s_pred_spans)
            whole_span: tuple[int, int] = (0, s_len - 1)
            f1_scores.append(f1_score(gold_spans, s_pred_spans_set, whole_span))
        f1_scores = torch.tensor(f1_scores, device=self.device)
        self.update_score(f1_scores)
        return f1_scores
