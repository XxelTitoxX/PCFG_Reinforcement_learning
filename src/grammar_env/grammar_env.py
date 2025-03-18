from collections import defaultdict
from logging import getLogger

import numpy as np

from grammar_env.criterion.criterion import Criterion
from grammar_env.grammar.binary_grammar import BinaryGrammarFactory
from grammar_env.grammar.unary_grammar import UnaryGrammar

__all__ = ['GrammarEnv']

logger = getLogger(__name__)


def _is_reachable(target: int, binary_rule_probs: list[tuple[int, int, int, float]]):
    """
    Check if the target symbol is reachable from the start symbol.
    :param target: int, the target symbol
    :param binary_rule_probs: list of tuples (lhs, rhs1, rhs2, prob) for binary rules lhs -> rhs1 rhs2
    :return: bool, True if the target symbol is reachable from the start symbol
    """
    # lhs symbols that can go to the rhs symbol
    rhs_to_lhs: defaultdict[int, list[int]] = defaultdict(list)
    for lhs, rhs1, rhs2, _ in binary_rule_probs:
        rhs_to_lhs[rhs1].append(lhs)
        rhs_to_lhs[rhs2].append(lhs)

    # Symbols that can go to the target symbol
    reachable_inverse: set[int] = set()
    stack: list[int] = [target]
    while stack:
        rhs: int = stack.pop()
        reachable_inverse.add(rhs)
        for lhs in rhs_to_lhs[rhs]:
            if lhs not in reachable_inverse:
                stack.append(lhs)
    return 0 in reachable_inverse


class GrammarEnv:
    def __init__(
            self, criterion: Criterion, max_productions: int,
            binary_grammar_factory: BinaryGrammarFactory,
            unary_grammar: UnaryGrammar
    ):
        self.criterion: Criterion = criterion
        self.max_productions: int = max_productions

        self.binary_grammar_factory: BinaryGrammarFactory = binary_grammar_factory
        self.unary_grammar: UnaryGrammar = unary_grammar

        self.num_nt: int = binary_grammar_factory.num_nt
        self.num_sym: int = binary_grammar_factory.num_sym
        self.num_rhs: int = binary_grammar_factory.num_rhs
        self.num_r: int = binary_grammar_factory.num_r

        self.state: np.ndarray = np.zeros(binary_grammar_factory.num_r, dtype=np.float32)
        self.score: float = 0.
        self.reset()

        logger.info(
            f"Environment initialized with {max_productions} max productions, "
            f"{binary_grammar_factory} binary grammar factory."
        )

    def _is_lhs_reachable(self, state: np.ndarray, action: tuple[int, float]) -> bool:
        production_idx, _ = action
        lhs_idx: int = production_idx // self.num_rhs
        if lhs_idx == 0:
            return True

        binary_rule_probs: list[tuple[int, int, int, float]] = self.binary_grammar_factory.create(
            self.state_to_reduced(state)
        ).rule_probs
        return _is_reachable(lhs_idx, binary_rule_probs)

    def step(self, action: tuple[int, float]) -> tuple[np.ndarray, float]:
        # is_lhs_reachable: bool = self._is_lhs_reachable(self.state, action)
        prev_score: float = self.score
        production_idx, production_weight = action
        self.state[production_idx] += production_weight

        cur_state_reduced: np.ndarray = self.state_to_reduced(self.state)
        cur_score: float = self.criterion.score(
            self.binary_grammar_factory.create(cur_state_reduced),
            self.unary_grammar
        )
        self.score = cur_score

        penalty: float = 0.
        # TODO: is this penalty helpful?
        # if not is_lhs_reachable:
        #     penalty += 0.2

        return (
            self.state.copy(), cur_score - prev_score - penalty
        )

    def reset(self) -> np.ndarray:
        """
        Reset the environment state and return the reduced form of the state.
        :return: np.ndarray of shape (n=0, 2), each row is [production, prob]
        """
        self.state[:] = 0.
        self.score = 0.
        self.criterion.refresh_dataloader()
        return self.state.copy()

    def state_to_reduced(self, state: np.ndarray) -> np.ndarray:
        """
        Convert the state to a reduced form of [[production0, prob0], [production1, prob1], ...].
        :param state: np.ndarray of shape (self.num_r,)
        :return: np.ndarray of shape (n, 2), each row is [production, prob]
                    where production is the index of the production rule, and prob is the probability
        """
        assert len(state.shape) == 1, f"state must be an 1d array, got {state.shape}"
        assert state.shape == (self.num_r,), (
            f"state must have a shape {self.num_r}, got {state.shape}"
        )
        assert state.dtype == np.float32, f"state.dtype must be np.float32, got {state.dtype}"

        state_weight: np.ndarray = state.reshape(self.num_nt, self.num_rhs)
        state_weight_sum: np.ndarray = state_weight.sum(axis=1)
        assert state_weight_sum.shape == (self.num_nt,)
        # If state_weight_sum is 0, set it to 1 to avoid division by zero.
        # The resulting state_prob will all be zeros anyway.
        state_weight_sum[state_weight_sum == 0] = 1
        state_prob: np.ndarray = state_weight / state_weight_sum[:, None]

        production_indexes: list[int] = []
        probabilities: list[float] = []
        for lhs_idx, rhs_idx in zip(*state_prob.nonzero()):
            production_indexes.append(lhs_idx * self.num_rhs + rhs_idx)
            probabilities.append(state_prob[lhs_idx, rhs_idx].item())

        return np.stack([
            np.array(production_indexes, dtype=np.float32),
            np.array(probabilities, dtype=np.float32)
        ], axis=-1)

    def is_endstate(self) -> bool:
        rules_num: int = self.state.sum().item()
        return rules_num >= self.max_productions
