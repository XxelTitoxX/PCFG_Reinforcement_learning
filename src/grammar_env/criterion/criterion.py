from abc import ABC, abstractmethod
from typing import Optional, final

import torch
from numpy.random import Generator, default_rng
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler

from grammar_env.corpus.corpus import Corpus
from grammar_env.grammar.binary_grammar import BinaryGrammar
from grammar_env.grammar.unary_grammar import UnaryGrammar

__all__ = ['Criterion']


class CorpusDataset(Dataset):
    def __init__(self, corpus: Corpus):
        self.corpus: Corpus = corpus
        self.sentences_idx: list[list[int]] = [
            # <unk> is mapped to index 1
            [corpus.symbol_to_idx.get(symbol, 1) for symbol in sentence.symbols]
            for sentence in corpus.sentences
        ]

    def __len__(self):
        return len(self.sentences_idx)

    def __getitem__(self, idx):
        return idx, self.sentences_idx[idx]


def collate_fn(batch: list[tuple[int, list[int]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indexes, sentences = zip(*batch)
    indexes: list[int]
    sentences: list[list[int]]
    return (
        torch.tensor(indexes, dtype=torch.long),
        pad_sequence(
            [torch.tensor(sentence, dtype=torch.long) for sentence in sentences],
            batch_first=True,
            # <pad> is mapped to index 0
            padding_value=0
        ), torch.tensor([len(sentence) for sentence in sentences], dtype=torch.long)
    )


def get_indexes(
        corpus_length: int, num_sentences_per_score: int, generator: Generator
) -> list[int]:
    return generator.choice(
        corpus_length, num_sentences_per_score, replace=False
    ).tolist()


class Criterion(ABC):
    def __init__(
            self, corpus: Corpus, device: torch.device,
            num_sentences_per_score: int, num_sentences_per_batch: int
    ):
        self.corpus: Corpus = corpus
        self.device: torch.device = device
        self.generator: Generator = default_rng(0)
        self.corpus_dataloader: Optional[DataLoader] = None

        self.num_sentences_per_score: int = min(num_sentences_per_score, len(corpus))
        self.num_sentences_per_batch: int = num_sentences_per_batch
        self._scores: torch.Tensor = torch.zeros(self.num_sentences_per_score, device=device)

        self.refresh_dataloader()
        assert self.corpus_dataloader is not None, "Corpus dataloader must be initialized"

        self.opt_binary_grammar: Optional[BinaryGrammar] = None
        """
        The PCFG with the highest score.
        If multiple PCFGs have the same score, the one that is scored last is selected.
        """
        self.opt_score: float = float("-inf")
        """
        Score of the optimal PCFG.
        i.e., the highest score.
        """

    def get_dataloader(
            self, num_sentences_per_score: int, num_sentences_per_batch: int
    ) -> DataLoader:
        if num_sentences_per_score >= len(self.corpus):
            sampler: Optional[Sampler] = None
        else:
            sampler: Optional[Sampler] = SubsetRandomSampler(
                get_indexes(
                    len(self.corpus), num_sentences_per_score,
                    self.generator
                )
            )
        return DataLoader(
            CorpusDataset(self.corpus),
            batch_size=num_sentences_per_batch,
            collate_fn=collate_fn,
            shuffle=False,
            sampler=sampler,
            pin_memory=True
        )

    @abstractmethod
    def score_sentence(
            self, binary_grammar: BinaryGrammar, unary_grammar: UnaryGrammar,
            sentence_indexes: torch.Tensor, sentences: torch.Tensor, sentence_lengths: torch.Tensor
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
    def refresh_dataloader(self):
        self.corpus_dataloader = self.get_dataloader(
            self.num_sentences_per_score, self.num_sentences_per_batch
        )

    @final
    def score(self, binary_grammar: BinaryGrammar, unary_grammar: UnaryGrammar) -> float:
        unary_grammar.eval()
        with torch.no_grad():
            scores_list: list[torch.Tensor] = [
                self.score_sentence(
                    binary_grammar, unary_grammar,
                    indexes.to(self.device), sentences.to(self.device),
                    lens.to(self.device)
                ) for indexes, sentences, lens in self.corpus_dataloader
            ]
            scores: torch.Tensor = torch.cat(scores_list, dim=0)
            self._scores = scores
            assert scores.shape == (self.num_sentences_per_score,), (
                f"Expected scores to have shape ({self.num_sentences_per_score},), got {scores.shape}"
            )
            score: float = torch.mean(scores).item()

        assert (not len(binary_grammar.rule_probs) == 0) or score == 0., (
            f"If binary_grammar {binary_grammar} is empty, score must be 0, got {score}"
        )

        if score > self.opt_score:
            self.opt_score = score
            self.opt_binary_grammar = binary_grammar
        return score
