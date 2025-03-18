from logging import getLogger

import torch

from grammar_env.corpus.corpus import Corpus
from grammar_env.criterion.criterion import Criterion
from grammar_env.criterion.inside_algorithm import log_probability_sentence_given_grammar
from grammar_env.grammar.binary_grammar import BinaryGrammar
from grammar_env.grammar.unary_grammar import UnaryGrammar

__all__ = ['ProbabilityCriterion']

logger = getLogger(__name__)


class ProbabilityCriterion(Criterion):
    def __init__(
            self, corpus: Corpus, device: torch.device, lower_bound: float,
            num_sentences_per_score: int, num_sentences_per_batch: int
    ):
        super().__init__(
            corpus, device,
            num_sentences_per_score, num_sentences_per_batch
        )

        assert lower_bound <= 0., f"lower_bound must be less than or equal to 0, got {lower_bound}"
        self.lower_bound: float = lower_bound

        logger.info(
            f"ProbabilityCriterion initialized with lower_bound={lower_bound:.5f}"
        )

    def probability_to_score(self, log_p: torch.Tensor) -> torch.Tensor:
        device: torch.device = log_p.device
        return torch.maximum(torch.tensor(0., device=device), log_p - self.lower_bound)

    def score_sentence(
            self, binary_grammar: BinaryGrammar, unary_grammar: UnaryGrammar,
            sentence_indexes: torch.Tensor, sentences: torch.Tensor, sentence_lengths: torch.Tensor
    ) -> torch.Tensor:
        log_p: torch.Tensor = log_probability_sentence_given_grammar(
            binary_grammar, unary_grammar,
            sentences, sentence_lengths
        )
        return self.probability_to_score(log_p)
