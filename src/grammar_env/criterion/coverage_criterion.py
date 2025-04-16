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
    ):
        super().__init__(
            corpus, device
        )

    def score_sentences(
            self, env
    ) -> torch.Tensor:
        scores = env.done.float()
        self.update_score(scores)
        return scores
