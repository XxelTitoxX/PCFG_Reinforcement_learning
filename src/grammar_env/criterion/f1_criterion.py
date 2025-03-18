import torch

from grammar_env.corpus.corpus import Corpus
from grammar_env.criterion.criterion import Criterion
from grammar_env.criterion.inside_algorithm import parse_sentences
from grammar_env.grammar.binary_grammar import BinaryGrammar
from grammar_env.grammar.unary_grammar import UnaryGrammar

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
            num_sentences_per_score: int, num_sentences_per_batch: int
    ):
        super().__init__(
            corpus, device,
            num_sentences_per_score, num_sentences_per_batch
        )

    def score_sentence(
            self, binary_grammar: BinaryGrammar, unary_grammar: UnaryGrammar,
            sentence_indexes: torch.Tensor, sentences: torch.Tensor, sentence_lengths: torch.Tensor
    ) -> torch.Tensor:
        spans: list[list[tuple[int, int, int]]] = parse_sentences(
            binary_grammar, unary_grammar,
            sentences, sentence_lengths
        )

        f1_scores: list[float] = []
        for b in range(len(sentence_indexes)):
            sentence_idx: int = sentence_indexes[b].item()
            gold_spans: set[tuple[int, int]] = set(
                (gold_span.start, gold_span.end)
                for gold_span in self.corpus.sentences[sentence_idx].gold_spans
            )
            pred_spans: set[tuple[int, int]] = set((i, j) for _, i, j in spans[b])
            whole_span: tuple[int, int] = (0, sentence_lengths[b].item() - 1)
            f1_scores.append(f1_score(gold_spans, pred_spans, whole_span))
        return torch.tensor(f1_scores, device=self.device) * 100.
