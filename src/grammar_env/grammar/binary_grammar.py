from dataclasses import dataclass

import numpy as np
from dataclasses_json import dataclass_json

__all__ = ['BinaryGrammar', 'BinaryGrammarFactory']


@dataclass_json
@dataclass(frozen=True)
class BinaryGrammar:
    num_nt: int
    """number of total non-terminals, |N| where N is a set of non-terminal symbols"""
    num_pt: int
    """number of total pre-terminal symbols, |P| where P is a set of pre-terminal symbols"""
    num_rhs: int
    """number of total right hand side terms, (|N| + |P|)^2"""
    rule_probs: list[tuple[int, int, int, float]]
    """list of rules X -> Y Z [prob]. i.e., list of tuples (X, Y, Z, prob)"""

    def __len__(self) -> int:
        return len(self.rule_probs)

    def __repr__(self):
        rule_probs_str: str = ", ".join(
            f"({lhs_idx}, {nt1_idx}, {nt2_idx}, {prob:.5f})"
            for lhs_idx, nt1_idx, nt2_idx, prob in self.rule_probs
        )
        return (
            f"BinaryGrammar("
            f"num_nt={self.num_nt}, num_pt={self.num_pt}, "
            f"num_rhs={self.num_rhs}, rule_len={len(self.rule_probs)}, "
            f"rule_probs=[{rule_probs_str}])"
        )


class BinaryGrammarFactory:
    def __init__(self, num_nt: int, num_pt: int):
        self.num_nt: int = num_nt
        """number of total non-terminals, |N| where N is a set of non-terminal symbols"""
        self.num_pt: int = num_pt
        """number of total pre-terminal symbols, |P| where P is a set of pre-terminal symbols"""
        self.num_sym: int = num_nt + num_pt
        """number of total symbols, |N| + |P|"""
        self.num_rhs: int = self.num_sym ** 2
        """number of total right hand side terms, (|N| + |P|)^2"""
        self.num_r: int = num_nt * self.num_rhs
        """number of total rules, |N| * (|N| + |P|)^2"""

    def __repr__(self):
        return (
            f"BinaryGrammarFactory("
            f"num_nt={self.num_nt}, num_pt={self.num_pt}, "
            f"num_rhs={self.num_rhs}, num_r={self.num_r})"
        )

    def get_rule_probs(self, state_reduced: np.ndarray) -> list[tuple[int, int, int, float]]:
        """
        Get the rules from the state_reduced form.
        :param state_reduced: np.ndarray of shape (n, 2), each row is [production, prob]
        :return: list of tuples (X, Y, Z, prob)
        """
        assert len(state_reduced.shape) == 2, f"state_reduced must be a 2d array, got {state_reduced.shape}"
        assert state_reduced.shape[1] == 2, f"state_reduced must have a shape (n, 2), got {state_reduced.shape}"
        assert state_reduced.dtype == np.float32, f"state_reduced.dtype must be np.float32, got {state_reduced.dtype}"

        productions_indexes: list[int] = state_reduced[:, 0].astype(np.int32).tolist()
        probabilities: list[float] = state_reduced[:, 1].tolist()

        rule_probs: list[tuple[int, int, int, float]] = []
        for production_idx, prob in zip(productions_indexes, probabilities):
            lhs_idx: int = production_idx // self.num_rhs
            rhs_idx: int = production_idx % self.num_rhs
            rhs1_idx: int = rhs_idx // self.num_sym
            rhs2_idx: int = rhs_idx % self.num_sym
            rule_probs.append((lhs_idx, rhs1_idx, rhs2_idx, prob))

        return rule_probs

    def create(self, state_reduced: np.ndarray) -> BinaryGrammar:
        """
        Create a BinaryGrammar from the state_reduced form.
        :param state_reduced: np.ndarray of shape (n, 2), each row is [production, prob]
        :return: BinaryGrammar
        """
        return BinaryGrammar(
            self.num_nt, self.num_pt, self.num_rhs, self.get_rule_probs(state_reduced)
        )
