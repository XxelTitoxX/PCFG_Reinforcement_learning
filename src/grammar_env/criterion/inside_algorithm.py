"""
Okay, I guess you figured out where our reinforcement learning's bottleneck is.
You're right.
The inside algorithm takes most of the time in our training loop.
I know you really want to optimize this code.
However, I've already put much work, and it'll be very hard to optimize it farther.

If you still want to give it a try, please remember three things:
1. Please understand pytorch and numba, and how to optimize them.
1. Make sure all the unit tests in test/test_inside_algorithm.py pass.
2. Check if the optimization really works by measuring the time.
"""

import time
from logging import getLogger

import numpy as np
import torch
from numba import njit

from grammar_env.grammar.binary_grammar import BinaryGrammar
from grammar_env.grammar.unary_grammar import UnaryGrammar

__all__ = ['log_probability_sentence_given_grammar', 'parse_sentences']

logger = getLogger(__name__)


def __calculate_dp(
        pi: torch.Tensor, binary_rule_probs: list[tuple[int, int, int, float]],
        maximum: bool
) -> None:
    """
    pi: torch.Tensor of shape (batch_size, num_symbols, n, n)
    binary_rule_probs: list of tuples (X, Y, Z, prob) for rules X -> Y Z
    maximum: if True, use max (for parsing); otherwise, use sum.
    """
    time_s: float = time.time()
    device: torch.device = pi.device
    batch_size: int = pi.shape[0]
    n: int = pi.shape[2]
    R: int = len(binary_rule_probs)
    R_full: int = pi.shape[1]

    # Convert binary_rule_probs into tensors.
    rule_X = torch.tensor([r[0] for r in binary_rule_probs], device=device, dtype=torch.long)
    rule_Y = torch.tensor([r[1] for r in binary_rule_probs], device=device, dtype=torch.long)
    rule_Z = torch.tensor([r[2] for r in binary_rule_probs], device=device, dtype=torch.long)
    rule_log_prob = torch.tensor(
        [r[3] for r in binary_rule_probs], device=device, dtype=pi.dtype
    ).log()

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            # Generate vector of split points: s in [i, j)
            L: int = j - i
            s_range = torch.arange(i, j, device=device)  # shape (L,)

            # left[b, r, s] = pi[b, rule_Y[r], i, s] for b in batches, r in rules, s in s_range
            left = pi[:, rule_Y[:, None], i, s_range[None, :]]
            assert left.shape == (batch_size, R, L)

            # right[b, r, s] = pi[b, rule_Z[r], s + 1, j] for b in batches, r in rules, s in s_range
            right = pi[:, rule_Z[:, None], s_range[None, :] + 1, j]
            assert right.shape == (batch_size, R, L)

            # e^pi(X, i, j)
            # = sum_{X -> YZ in R, s in [i, j - 1]} (e^log P(X -> AB) * e^pi(Y, i, s) * e^pi(Z, s + 1, j))
            # pi(X, i, j)
            # = logsumexp_{X -> YZ in R, s in [i, j - 1]} (log P(X -> AB) + pi(Y, i, s) + pi(Z, s + 1, j))
            # If maximum is true, use max instead of sum (used for parsing)
            mul_prob_lr: torch.Tensor = left + right
            mul_prob_lr += rule_log_prob[None, :, None]
            assert mul_prob_lr.shape == (batch_size, R, L)

            # Rules can contain two rules with the same X.
            # Thus, rule_X can contain duplicate values.
            # We aggregate the values in aggregated_full using index_reduce
            if maximum:
                aggregated: torch.Tensor = mul_prob_lr.amax(dim=2)
                assert aggregated.shape == (batch_size, R)

                aggregated_full: torch.Tensor = torch.full(
                    (batch_size, R_full), -torch.inf, device=device
                )
                aggregated_full = aggregated_full.index_reduce(dim=1, index=rule_X, source=aggregated, reduce="amax")
            else:
                aggregated: torch.Tensor = mul_prob_lr.logsumexp(dim=2)
                assert aggregated.shape == (batch_size, R)

                _logsumexp_m: torch.Tensor = aggregated.amax(dim=1)
                # As -inf - (-inf) = nan, we replace -inf with 0
                _logsumexp_m[_logsumexp_m == -float('inf')] = 0
                aggregated -= _logsumexp_m[:, None]
                aggregated_full: torch.Tensor = torch.zeros(
                    (batch_size, R_full), device=device
                )
                aggregated_full = aggregated_full.index_add(dim=1, index=rule_X, source=aggregated.exp())
                torch.log(aggregated_full, out=aggregated_full)
                aggregated_full += _logsumexp_m[:, None]

            # Save the aggregated values to pi[:, :, i, j]
            pi[:, :, i, j] = aggregated_full

    logger.debug(
        f"DP calculation time: {time.time() - time_s:.5f} seconds, "
        f"batch: {batch_size}, n: {n}, num_rules: {len(binary_rule_probs)}"
    )


def __inside_algorithm(
        binary_grammar: BinaryGrammar, unary_grammar: UnaryGrammar,
        sentences: torch.Tensor, maximum: bool = False
) -> torch.Tensor:
    """
    Calculate pi of the inside algorithm.
    :param binary_grammar: Binary part of the PCFG
    :param unary_grammar: Unary part of the PCFG
    :param sentences: torch.Tensor of shape (batch_size, max_sentence_length)
    :param maximum: If True, use max instead of sum (used for parsing)
    :return: pi of the inside algorithm
            torch.Tensor of shape (batch_size, num_nt + num_pt, max_sentence_length, max_sentence_length)
    """
    device: torch.device = sentences.device

    batch_size, n = sentences.shape
    # n is the maximum sentence length
    num_nt, num_pt = binary_grammar.num_nt, binary_grammar.num_pt

    unary_rule_log_probs: torch.Tensor = unary_grammar.get_rule_log_probs(sentences)

    # pi[b, X, i, j] is the log probability of generating a subsentence x_i, x_i+1, ..., x_j of sentences[b],
    # given a non-terminal or pre-terminal X as the root
    pi: torch.Tensor = torch.full(
        (batch_size, num_nt + num_pt, n, n), -torch.inf, device=device
    )

    # Initialize pi[b, X, i, i] for all pre-terminals X
    # pi[b, X, i, i] = log (q(X -> x_i) if X -> x_i in PCFG else 0)
    # In pi, X is a set of non-terminal or pre-terminal symbols (index: [0, num_nt + num_pt))
    # Non-terminal is indexed from 0, and pre-terminal is indexed from num_nt
    # In unary_rule_log_probs, X is a set of pre-terminal symbols (index: [0, num_pt))
    n_range: torch.Tensor = torch.arange(n, device=device)
    pi[:, num_nt:, n_range, n_range] = unary_rule_log_probs

    binary_rule_probs: list[tuple[int, int, int, float]] = binary_grammar.rule_probs
    if len(binary_rule_probs) > 0:
        __calculate_dp(
            pi, binary_rule_probs, maximum
        )

    assert pi.isnan().sum().item() == 0, (
        f"pi must not have NaN, got {pi} with {pi.isnan().sum().item()} NaN"
    )
    return pi


def log_probability_sentence_given_grammar(
        binary_grammar: BinaryGrammar, unary_grammar: UnaryGrammar,
        sentences: torch.Tensor, sentence_lengths: torch.Tensor
) -> torch.Tensor:
    """
    Calculate log P(sentence | PCFG), the probability of a sentence given a PCFG.
    :param binary_grammar: Binary part of the PCFG
    :param unary_grammar: Unary part of the PCFG
    :param sentences: torch.Tensor of shape (batch_size, max_sentence_length)
    :param sentence_lengths: torch.Tensor of shape (batch_size,)
    :return: log P(sentence | PCFG)
            torch.Tensor of shape (batch_size,)
    """
    batch_size: int = sentences.shape[0]
    device: torch.device = sentences.device
    pi: torch.Tensor = __inside_algorithm(
        binary_grammar, unary_grammar, sentences, maximum=False
    )

    # The probability of generating the whole sentence sentences[b]
    # is pi[b, S, 0, len(sentence[b]) - 1] for b in batch
    return pi[torch.arange(batch_size, device=device), 0, 0, sentence_lengths - 1]


#@njit
def __backtrack_one(
        pi: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, log_prob: np.ndarray,
        nt: int, i: int, j: int
) -> tuple[int, int]:
    R: int = X.shape[0]
    for r in range(R):
        if X[r] == nt:
            for s in range(i, j):
                if np.isclose(pi[nt, i, j], pi[Y[r], i, s] + pi[Z[r], s + 1, j] + log_prob[r]):
                    return r, s

    assert False, (
        f"Backtrack failed for nt={nt}, i={i}, j={j}, pi[nt, i, j]: {pi[nt, i, j]}, "
        f"pi={pi[:, i:j + 1, i:j + 1]}, X: {X}, Y: {Y}, Z: {Z}, prob: {log_prob}"
    )


# @njit
# def __backtrack(
#         pi: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, log_prob: np.ndarray,
#         nt: int, i: int, j: int
# ) -> list[tuple[int, int, int]]:
#     """
#     Backtrack the parse tree.
#     :param pi: pi of the inside algorithm
#             np.ndarray of shape (num_nt + num_pt, max_sentence_length, max_sentence_length)
#     :param X: non-terminal index X of the rule X -> Y Z [prob]
#     :param Y: non-terminal index Y of the rule X -> Y Z [prob]
#     :param Z: non-terminal index Z of the rule X -> Y Z [prob]
#     :param log_prob: probability of the rule X -> Y Z [prob]
#     :param nt: non-terminal index to backtrack
#     :param i: left sentence index to backtrack
#     :param j: right sentence index to backtrack
#     :return: list of spans of the parse tree
#     """
#     assert len(pi.shape) == 3, f"pi must be a 3d array, got {pi.shape}"
#     assert i <= j, f"i must be less than or equal to j, got i={i}, j={j}"

#     spans: list[tuple[int, int, int]] = []
#     stack: list[tuple[int, int, int]] = [(nt, i, j)]
#     while stack:
#         nt, i, j = stack.pop()
#         if i == j:
#             continue
#         if pi[nt, i, j].item() == float('-inf'):
#             continue

#         r, s =  __backtrack_one(pi, X, Y, Z, log_prob, nt, i, j)

#         spans.append((nt, i, j))
#         stack.append((Z[r].item(), s + 1, j))
#         stack.append((Y[r].item(), i, s))

#     return spans

#@njit
def __backtrack(
    pi: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, log_prob: np.ndarray,
    nt: int, i: int, j: int
) -> list[tuple[int, int, int]]:
    """
    Backtrack the parse tree.
    """
    assert len(pi.shape) == 3, f"pi must be a 3d array, got {pi.shape}"
    assert i <= j, f"i must be less than or equal to j, got i={i}, j={j}"
    spans = []
    stack = [(nt, i, j)]
    
    # Define negative infinity as a constant that Numba can handle
    NEG_INF = np.NINF  # Use NumPy's negative infinity constant
    
    while stack:
        nt, i, j = stack.pop()
        if i == j:
            continue
        # Compare with the NumPy constant instead of using float('-inf')
        if pi[nt, i, j] == NEG_INF:
            continue
        r, s = __backtrack_one(pi, X, Y, Z, log_prob, nt, i, j)
        spans.append((nt, i, j))
        stack.append((int(Z[r]), s + 1, j))
        stack.append((int(Y[r]), i, s))
    return spans


def parse_sentences(
        binary_grammar: BinaryGrammar, unary_grammar: UnaryGrammar,
        sentences: torch.Tensor, sentence_lengths: torch.Tensor
) -> list[list[tuple[int, int, int]]]:
    """
    Parse sentences using the inside algorithm.
    :param binary_grammar: Binary part of the PCFG
    :param unary_grammar: Unary part of the PCFG
    :param sentences: torch.Tensor of shape (batch_size, max_sentence_length)
    :param sentence_lengths: torch.Tensor of shape (batch_size,)
    :return: list of (list of tuples), each tuple is a span represented by (nt_idx, span_left, span_right)
    """
    pi: torch.Tensor = __inside_algorithm(
        binary_grammar, unary_grammar, sentences, maximum=True
    )

    time_s: float = time.time()
    batch_size: int = sentences.shape[0]
    pi_np: np.ndarray = pi.numpy(force=True)
    X: np.ndarray = np.array([r[0] for r in binary_grammar.rule_probs], dtype=np.int64)
    Y: np.ndarray = np.array([r[1] for r in binary_grammar.rule_probs], dtype=np.int64)
    Z: np.ndarray = np.array([r[2] for r in binary_grammar.rule_probs], dtype=np.int64)
    log_prob: np.ndarray = np.log(np.array([r[3] for r in binary_grammar.rule_probs], dtype=np.float32))

    spans_batch: list[list[tuple[int, int, int]]] = []
    for b in range(batch_size):
        spans: list[tuple[int, int, int]] = __backtrack(
            pi_np[b, :, :, :], X, Y, Z, log_prob,
            0, 0, sentence_lengths[b].item() - 1
        )
        spans_batch.append(spans)

    logger.debug(
        f"Backtrack time: {time.time() - time_s:.5f} seconds, "
        f"batch: {batch_size}, n: {sentences.shape[1]}, num_rules: {len(binary_grammar.rule_probs)}"
    )
    return spans_batch
