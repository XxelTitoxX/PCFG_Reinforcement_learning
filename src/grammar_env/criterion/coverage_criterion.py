import torch

from grammar_env.criterion.criterion import Criterion
from env import Environment

__all__ = ['CoverageCriterion']


class CoverageCriterion(Criterion):
    def __init__(
            self, device: torch.device,
    ):
        super().__init__(
            device
        )

    def score_sentences(
            self, env : Environment
    ) -> torch.Tensor:
        scores = (env.success()).float()
        self.update_score(scores)
        return scores
