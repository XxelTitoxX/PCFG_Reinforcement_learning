from abc import ABC, abstractmethod
from typing import Optional, final

import numpy as np
import torch

from grammar_env.corpus.corpus import Corpus
from actor_critic import ActorCritic

__all__ = ['Criterion']



class Criterion(ABC):
    def __init__(
            self, corpus: Corpus, device: torch.device
    ):
        self.corpus: Corpus = corpus
        self.device: torch.device = device

        self.opt_model: Optional[ActorCritic] = None
        """
        The PCFG with the highest score.
        If multiple PCFGs have the same score, the one that is scored last is selected.
        """
        self.opt_score: float = float("-inf")
        """
        Score of the optimal PCFG.
        i.e., the highest score.
        """
        self.last_score: float = float("-inf")

    @abstractmethod
    def score_sentences(
            self, env
    ) -> torch.Tensor:
        """
        Score of sentences given the PCFG.
        Score given an empty PCFG is 0.
        :param binary_grammar: Binary part of the PCFG
        :param unary_grammar: Unary part of the PCFG
        :param sentence_indexes: torch.Tensor of shape (batch_size,),
                                indexes of the sentences (in the corpus)
        :param sentences: torch.Tensor of shape (batch_size, max_sentence_length)
        :param sentence_lengths: torch.Tensor of shape (batch_size,)
        """
        pass

    @final
    def update_score(self, scores : torch.Tensor):
        self.last_score = torch.mean(scores).item()
        if self.last_score > self.opt_score:
            self.opt_score = self.last_score

    def update_optimal_model(
            self, model: ActorCritic):
        if self.last_score == self.opt_score:
            self.opt_model = model.copy()
