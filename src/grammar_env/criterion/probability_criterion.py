from logging import getLogger

import torch
import numpy as np

from grammar_env.corpus.corpus import Corpus
from grammar_env.criterion.criterion import Criterion
from env import Environment

__all__ = ['ProbabilityCriterion']

logger = getLogger(__name__)


class ProbabilityCriterion(Criterion):
    def __init__(
            self, corpus: Corpus, device: torch.device
    ):
        super().__init__(
            corpus, device
        )

    def score_sentences(
        self, env: Environment
) -> torch.Tensor:
        sentence_parsing_log_probs = torch.sum(env.done[:, None] * env.rew * env.actions_log_probs, dim=1) + (~env.done) * float('-inf')
        parsing_probs = torch.exp(sentence_parsing_log_probs)
        self.update_score(parsing_probs)
        return parsing_probs
