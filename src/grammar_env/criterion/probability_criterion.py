from logging import getLogger

import torch

from grammar_env.criterion.criterion import Criterion
from env import Environment

__all__ = ['ProbabilityCriterion']

logger = getLogger(__name__)


class ProbabilityCriterion(Criterion):
    def __init__(
            self, device: torch.device
    ):
        super().__init__(
            device
        )

    def score_sentences(
        self, env: Environment
) -> torch.Tensor:
        log_probs = env.positions_log_probs + env.symbols_log_probs
        batch_size, max_len = log_probs.shape
        mask = torch.arange(max_len, device=env.device).unsqueeze(0).expand(batch_size, -1) >= env.ep_len.unsqueeze(1)
        log_probs[mask] = 0.

        not_parsed_probs = torch.zeros(batch_size, device=env.device)
        not_parsed_probs[~env.done] = float('-inf')
        sentence_parsing_log_probs = torch.sum(log_probs, dim=1) + not_parsed_probs
        parsing_probs = torch.exp(sentence_parsing_log_probs)
        self.update_score(parsing_probs)
        return parsing_probs
