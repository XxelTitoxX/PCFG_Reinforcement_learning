from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
from typing_extensions import final

from grammar_env.corpus.action import Shift
from grammar_env.corpus.corpus import Corpus

__all__ = ['UnaryGrammar', 'DeterminedUnaryGrammar', 'SupervisedUnaryGrammar']


class UnaryGrammar(nn.Module, ABC):
    def __init__(self, num_pt: int, num_t: int):
        super().__init__()

        self.num_pt: int = num_pt
        self.num_t: int = num_t

    @abstractmethod
    def _get_rule_log_probs(self, sentences: torch.Tensor) -> torch.Tensor:
        pass

    @final
    def get_rule_log_probs(self, sentences: torch.Tensor) -> torch.Tensor:
        """
        Return the log probability of each pre-terminal generating each terminal of the sentences.
        i.e., returned[batch_idx, pt_idx, t_idx] is the log probability of pt -> sentence[batch_idx, t_idx].
        :param sentences: torch.Tensor of shape (batch_size, max_sentence_length)
        :return: torch.Tensor of shape (batch_size, num_pt, max_sentence_length)
        """
        rule_log_probs = self._get_rule_log_probs(sentences)
        assert rule_log_probs.shape == (sentences.shape[0], self.num_pt, sentences.shape[1])
        return rule_log_probs


class DeterminedUnaryGrammar(UnaryGrammar):
    """
    A unary grammar with fixed rules. (no learning)
    """

    def __init__(self, num_pt: int, num_t: int, rules: torch.Tensor):
        """
        Initialize a unary grammar with the given rules.
        :param num_pt: Number of pre-terminals
        :param num_t: Number of terminals
        :param rules: Unary grammar rules.
                    The log probability for pt -> t is rules[pt, t].
        """
        super().__init__(num_pt, num_t)

        assert rules.shape == (num_pt, num_t), f"Expected shape ({num_pt}, {num_t}), got {rules.shape}"
        self.register_buffer('rules', rules)

    def _get_rule_log_probs(self, sentences: torch.Tensor) -> torch.Tensor:
        batch_size, max_sentence_length = sentences.shape
        num_pt: int = self.num_pt

        # Expand self.rules to shape (batch_size, num_pt, num_t)
        rules_expanded = self.rules.unsqueeze(0).expand(batch_size, num_pt, self.num_t)
        # Expand sentences to shape (batch_size, num_pt, max_sentence_length)
        sentences_expanded = sentences.unsqueeze(1).expand(batch_size, num_pt, max_sentence_length)

        # Gather the rule log probabilities for each terminal in the sentences
        rule_log_probs = torch.gather(rules_expanded, 2, sentences_expanded)
        assert rule_log_probs.shape == (batch_size, num_pt, max_sentence_length)

        return rule_log_probs


class SupervisedUnaryGrammar(DeterminedUnaryGrammar):
    def __init__(self, corpus: Corpus):
        pos_symbol_num: defaultdict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
        # Count the number of times each POS tag appears with each symbol

        pos_cluster: dict[str, str] = {
            "CC": "CC",

            "CD": "CD",

            "DT": "DT",
            "PDT": "DT",

            "IN": "IN",

            "JJ": "JJ",
            "JJR": "JJ",
            "JJS": "JJ",

            "NN": "NN",
            "NNS": "NN",
            "NNP": "NN",
            "NNPS": "NN",

            "PRP": "PRP",
            "PRP$": "PRP",

            "RB": "RB",
            "RBR": "RB",
            "RBS": "RB",

            "TO": "TO",

            "VB": "VB",
            "VBD": "VB",
            "VBG": "VB",
            "VBN": "VB",
            "VBP": "VB",
            "VBZ": "VB",

            "WDT": "WH-",
            "WP": "WH-",
            "WP$": "WH-",
            "WRB": "WH-",

            "MD": "MD",

            "POS": "POS",

            "EX": "ETC",
            "FW": "ETC",
            "LS": "ETC",
            "RP": "ETC",
            "SYM": "ETC",
            "UH": "ETC",
        }

        for sentence in corpus.sentences:
            for action in sentence.actions_sanitized:
                match action:
                    case Shift(pos_tag, symbol):
                        symbol_idx: int = corpus.symbol_to_idx.get(symbol, 1)
                        pos_symbol_num[pos_cluster[pos_tag]][symbol_idx] += 1
                    case _:
                        pass

        self.pos_to_idx: dict[str, int] = {}
        self.idx_to_pos: dict[int, str] = {}
        for idx, pos in enumerate(sorted(pos_symbol_num.keys())):
            self.pos_to_idx[pos] = idx
            self.idx_to_pos[idx] = pos

        num_pt: int = len(self.pos_to_idx)
        num_t: int = len(corpus.symbol_to_idx)

        rules: torch.Tensor = torch.zeros(num_pt, num_t)
        """
        Unary grammar rules.
        The log probability for pt -> t is rules[pt, t].
        Pre-terminal pt is the index of the POS tag, and terminal t is the index of the terminal symbol.
        """
        for pos, symbol_num in pos_symbol_num.items():
            pos_idx: int = self.pos_to_idx[pos]
            total_num: int = sum(symbol_num.values())
            for symbol_idx, num in symbol_num.items():
                rules[pos_idx, symbol_idx] = num / total_num

        super().__init__(num_pt, num_t, rules.log())
