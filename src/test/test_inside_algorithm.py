import math
import unittest

import torch

from grammar_env.criterion.inside_algorithm import log_probability_sentence_given_grammar, parse_sentences
from grammar_env.grammar.binary_grammar import BinaryGrammar
from grammar_env.grammar.unary_grammar import DeterminedUnaryGrammar


class TestLogProbability(unittest.TestCase):
    def test_pcfg1(self):
        """Test PCFG:
        0 -> 1 1 [1.0]
        1 -> 'a' [1.0]
        """
        # Binary grammar setup
        binary_rules = [(0, 1, 1, 1.0)]  # 0 -> 1 1
        binary_grammar = BinaryGrammar(num_nt=1, num_pt=1, num_rhs=4, rule_probs=binary_rules)

        # Unary grammar setup (1 pre-terminal -> 'a' with prob 1.0)
        rules = torch.tensor([[math.log(1.0)]], dtype=torch.float32)  # shape: (1, 1)
        unary_grammar = DeterminedUnaryGrammar(num_pt=1, num_t=1, rules=rules)

        # Test cases
        test_cases = [
            # (sentence, length, expected_log_prob)
            (torch.tensor([[0]]), torch.tensor([1]), -math.inf),  # "a" impossible
            (torch.tensor([[0, 0]]), torch.tensor([2]), math.log(1.0 * 1.0 * 1.0)),  # "aa" is always generated
            (torch.tensor([[0, 0, 0, 0]]), torch.tensor([4]), -math.inf),  # "aaaa" is impossible
        ]

        for sentences, lengths, expected in test_cases:
            log_prob = log_probability_sentence_given_grammar(
                binary_grammar, unary_grammar, sentences, lengths
            )
            self.assertEqual(log_prob.shape, (1,))
            self.assertAlmostEqual(log_prob.item(), expected, places=5)

    def test_pcfg2(self):
        """Test PCFG:
        0 -> 0 2 [0.5] | 2 2 [0.5]
        2 -> 'a' [1.0]
        """
        # Binary grammar setup
        binary_rules = [
            (0, 0, 2, 0.5),  # 0 -> 0 2
            (0, 2, 2, 0.5)  # 0 -> 2 2
        ]
        binary_grammar = BinaryGrammar(num_nt=2, num_pt=1, num_rhs=9, rule_probs=binary_rules)

        # Unary grammar setup (2 -> 'a' with prob 1.0)
        rules = torch.tensor([[math.log(1.0)]], dtype=torch.float32)
        unary_grammar = DeterminedUnaryGrammar(num_pt=1, num_t=1, rules=rules)

        test_cases = [
            (torch.tensor([[0]]), torch.tensor([1]), -math.inf),  # "a" impossible
            (torch.tensor([[0, 0]]), torch.tensor([2]), math.log(0.5 * 1.0 * 1.0)),  # "aa"
            (
                torch.tensor([[0, 0, 0]]), torch.tensor([3]),
                math.log(0.5 * (0.5 * 1.0 * 1.0 * 1.0))  # "aaa" is generated as (aa)a
            )
        ]

        for sentences, lengths, expected in test_cases:
            log_prob = log_probability_sentence_given_grammar(
                binary_grammar, unary_grammar, sentences, lengths
            )
            self.assertEqual(log_prob.shape, (1,))
            self.assertAlmostEqual(log_prob.item(), expected, places=5)

    def test_pcfg3(self):
        """Test PCFG:
        0 -> 0 0 [0.5] | 1 1 [0.5]
        1 -> 'a' [0.5] | 'b' [0.25] | 'c' [0.25]
        """
        # Binary grammar setup
        binary_rules = [
            (0, 0, 0, 0.5),  # 0 -> 0 0
            (0, 1, 1, 0.5)  # 0 -> 1 1
        ]
        binary_grammar = BinaryGrammar(num_nt=1, num_pt=1, num_rhs=4, rule_probs=binary_rules)

        # Unary grammar setup for terminal probabilities
        rules = torch.tensor([[math.log(0.5), math.log(0.25), math.log(0.25)]], dtype=torch.float32)
        unary_grammar = DeterminedUnaryGrammar(num_pt=1, num_t=3, rules=rules)

        test_cases = [
            (torch.tensor([[0, 0]]), torch.tensor([2]), math.log(0.5 * 0.5 * 0.5)),  # "aa"
            (torch.tensor([[1, 1]]), torch.tensor([2]), math.log(0.5 * 0.25 * 0.25)),  # "bb"
            (torch.tensor([[0, 1]]), torch.tensor([2]), math.log(0.5 * 0.5 * 0.25)),  # "ab"
            (
                torch.tensor([[2, 2, 2, 2, 2, 2]]), torch.tensor([6]),
                math.log((0.5 ** 5) * (0.25 ** 6) * 2)  # "cccccc" is generated as ((cc)(cc))(cc) or (cc)((cc)(cc))
            ),

        ]

        for sentences, lengths, expected in test_cases:
            log_prob = log_probability_sentence_given_grammar(
                binary_grammar, unary_grammar, sentences, lengths
            )
            self.assertEqual(log_prob.shape, (1,))
            self.assertAlmostEqual(log_prob.item(), expected, places=5)


class TestParseSentences(unittest.TestCase):
    def test_parse_pcfg1(self):
        """Test parsing with PCFG:
        0 -> 1 1 [1.0]
        1 -> 'a' [1.0]
        """
        # Binary grammar setup
        binary_rules = [(0, 1, 1, 1.0)]  # 0 -> 1 1
        binary_grammar = BinaryGrammar(num_nt=1, num_pt=1, num_rhs=4, rule_probs=binary_rules)

        # Unary grammar setup (1 pre-terminal -> 'a' with prob 1.0)
        rules = torch.tensor([[math.log(1.0)]], dtype=torch.float32)
        unary_grammar = DeterminedUnaryGrammar(num_pt=1, num_t=1, rules=rules)

        # Test cases
        test_cases = [
            # (sentence, length, expected_spans)
            (torch.tensor([[0]]), torch.tensor([1]), [[]]),  # "a" impossible
            (
                torch.tensor([[0, 0]]), torch.tensor([2]),
                [[(0, 0, 1)]]  # Parse tree for "aa": (0 (1 a) (1 a))
            ),
            (torch.tensor([[0, 0, 0, 0]]), torch.tensor([4]), [[]]),  # "aaaa" is impossible
        ]

        for sentences, lengths, expected_spans in test_cases:
            spans = parse_sentences(
                binary_grammar, unary_grammar,
                sentences, lengths
            )
            self.assertEqual(spans, expected_spans)

    def test_parse_pcfg2(self):
        """Test parsing with PCFG:
        0 -> 0 2 [0.5] | 2 2 [0.5]
        2 -> 'a' [1.0]
        """
        binary_rules = [
            (0, 0, 2, 0.5),  # 0 -> 0 2
            (0, 2, 2, 0.5)  # 0 -> 2 2
        ]
        binary_grammar = BinaryGrammar(num_nt=1, num_pt=2, num_rhs=9, rule_probs=binary_rules)

        rules = torch.tensor([[math.log(1.0)]], dtype=torch.float32)
        unary_grammar = DeterminedUnaryGrammar(num_pt=1, num_t=1, rules=rules)

        test_cases = [
            (torch.tensor([[0]]), torch.tensor([1]), [[]]),  # "a" impossible
            (
                torch.tensor([[0, 0]]), torch.tensor([2]),
                [[(0, 0, 1)]]  # Parse tree for "aa": (0 (2 a) (2 a))
            ),
            (
                torch.tensor([[0, 0, 0]]), torch.tensor([3]),
                [[(0, 0, 2), (0, 0, 1)]]
                # Parse tree for "aaa": (0 (0 (2 a) (2 a)) (2 a))
            )
        ]

        for sentences, lengths, expected_spans in test_cases:
            spans = parse_sentences(
                binary_grammar, unary_grammar,
                sentences, lengths
            )
            self.assertEqual(spans, expected_spans)

    def test_parse_pcfg3(self):
        """Test parsing with PCFG:
        0 -> 0 0 [0.5] | 1 1 [0.5]
        1 -> 'a' [0.5] | 'b' [0.25] | 'c' [0.25]
        """
        binary_rules = [
            (0, 0, 0, 0.5),  # 0 -> 0 0
            (0, 1, 1, 0.5)  # 0 -> 1 1
        ]
        binary_grammar = BinaryGrammar(num_nt=1, num_pt=1, num_rhs=4, rule_probs=binary_rules)

        rules = torch.tensor([[math.log(0.5), math.log(0.25), math.log(0.25)]], dtype=torch.float32)
        unary_grammar = DeterminedUnaryGrammar(num_pt=1, num_t=3, rules=rules)

        test_cases = [
            (
                torch.tensor([[0, 0]]), torch.tensor([2]),
                [[(0, 0, 1)]]  # Parse tree for "aa": (0 (1 a) (1 a))
            ),
            (
                torch.tensor([[2, 2, 2, 2, 2, 2]]), torch.tensor([6]),
                [[(0, 0, 5), (0, 0, 1), (0, 2, 5), (0, 2, 3), (0, 4, 5)]]
                # Parse tree for "cccccc": (0 (0 (1 c) (1 c)) (0 (0 (1 c) (1 c)) (0 (1 c) (1 c)))
                # There are two possible parse trees, but the first one should be returned
            )
        ]

        for sentences, lengths, expected_spans in test_cases:
            spans = parse_sentences(
                binary_grammar, unary_grammar,
                sentences, lengths
            )
            self.assertEqual(spans, expected_spans)


if __name__ == '__main__':
    unittest.main()
