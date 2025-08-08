import torch
from pathlib import Path
from typing import Optional

from grammar_env.criterion.criterion import Criterion
from env import Environment

__all__ = ['CoverageCriterion']


class CoverageCriterion(Criterion):
    def __init__(
            self, device: torch.device, persistent_dir: Optional[Path] = None
    ):
        super().__init__(
            device, persistent_dir
        )

    def score_sentences(
            self, env : Environment
    ) -> torch.Tensor:
        scores = (env.success()).float()
        self.update_score(scores.mean().item())
        return scores
