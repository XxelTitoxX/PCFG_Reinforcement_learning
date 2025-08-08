from abc import ABC, abstractmethod
from typing import final

import torch

__all__ = ['Criterion']



class Criterion(ABC):
    def __init__(
            self, device: torch.device
    ):
        self.device: torch.device = device
        self.opt_score: float = float("-inf")
        """
        Highest score achieved on a batch of sentences.
        """
        self.last_score: float = float("-inf")
        """
        Score achieved on the last batch of sentences.
        """

    @abstractmethod
    def score_sentences(
            self, env
    ) -> torch.Tensor:
        """
        Score of the bottom-up parsing of the batch of sentences in the environment.
        :param env: Environment with sentences to score.
        """
        pass

    @final
    def update_score(self, score : float):
        self.last_score = score
        if self.last_score > self.opt_score:
            self.opt_score = self.last_score

