import torch

from grammar_env.corpus.corpus import Corpus
from grammar_env.criterion.criterion import Criterion
from grammar_env.criterion.inside_algorithm import log_probability_sentence_given_grammar
from grammar_env.grammar.binary_grammar import BinaryGrammar
from grammar_env.grammar.unary_grammar import UnaryGrammar

__all__ = ['CoverageCriterion']


class CoverageCriterion(Criterion):
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
        p: torch.Tensor = log_probability_sentence_given_grammar(
            binary_grammar, unary_grammar,
            sentences, sentence_lengths
        )
        return (p > -torch.inf).float() * 100.
