import unittest
from pathlib import Path

import torch

from grammar_env.corpus.corpus import Corpus
from grammar_env.grammar.unary_grammar import SupervisedUnaryGrammar


class TestSupervisedUnaryGrammar(unittest.TestCase):
    def setUp(self):
        # Create test corpus from ptb_small.txt
        self.ptb_path: Path = Path('resources/ptb_small.txt')
        self.corpus = Corpus(str(self.ptb_path), max_vocab_size=3)
        self.grammar: SupervisedUnaryGrammar = SupervisedUnaryGrammar(self.corpus)

        # Expected POS tags from the sample sentence
        self.expected_pos_tags: set[str] = {
            "RB", "PRP", "VB", "IN", "DT", "NN"
        }

        # Expected terminals from the sample sentence
        # Note: symbols are sanitized (lowercase, numbers replaced with <num>)
        self.expected_terminals: set[str] = {
            "<pad>", "<unk>", "<num>", "now", "the"
        }

    def test_initialization(self):
        """Test if grammar is initialized correctly with expected POS tags and terminals"""
        # Check if all expected POS tags are present
        actual_pos_tags = set(self.grammar.pos_to_idx.keys())
        self.assertEqual(actual_pos_tags, self.expected_pos_tags)

        # Verify bidirectional mapping between POS tags and indices
        for pos, idx in self.grammar.pos_to_idx.items():
            self.assertEqual(self.grammar.idx_to_pos[idx], pos)

        # Check if all terminals are present
        self.assertEqual(set(self.corpus.symbol_to_idx.keys()), self.expected_terminals)

    def test_rule_probabilities(self):
        """Test if rule probabilities are calculated correctly"""
        # Create a test sentence tensor
        test_symbols = ["now", "hello", "the", "<num>", "unknown symbol"]
        test_indices = [self.corpus.symbol_to_idx.get(s, 1) for s in test_symbols]
        test_sentence = torch.tensor([test_indices])

        # Get rule probabilities
        rule_probs = self.grammar.get_rule_log_probs(test_sentence)

        # Check shape
        self.assertEqual(
            rule_probs.shape,
            (1, self.grammar.num_pt, len(test_symbols))
        )

        # Test specific rules

        # RB -> test_symbols[0] (="now") should have probability 1.0
        rb_idx = self.grammar.pos_to_idx["RB"]
        prob_rb_now = rule_probs[0, rb_idx, 0].exp().item()
        self.assertAlmostEqual(prob_rb_now, 1.0, places=5)

        # RB -> test_symbols[1] (="hello") should have probability 0.0
        prob_rb_hello = rule_probs[0, rb_idx, 1].exp().item()
        self.assertAlmostEqual(prob_rb_hello, 0.0, places=5)

        # NN -> test_symbols[2] (="the") should have probability 0.5
        nn_idx = self.grammar.pos_to_idx["NN"]
        prob_nn_the = rule_probs[0, nn_idx, 2].exp().item()
        self.assertAlmostEqual(prob_nn_the, 0.5, places=5)

        # IN -> test_symbols[3] (="<num>") should have probability 0.5
        in_idx = self.grammar.pos_to_idx["IN"]
        prob_in_num = rule_probs[0, in_idx, 3].exp().item()
        self.assertAlmostEqual(prob_in_num, 0.5, places=5)

        # IN -> test_symbols[4] (="<unk>") should have probability 0.5
        prob_in_unk = rule_probs[0, in_idx, 4].exp().item()
        self.assertAlmostEqual(prob_in_unk, 0.5, places=5)


if __name__ == '__main__':
    unittest.main()
