import unittest
from pathlib import Path


class TestAction(unittest.TestCase):
    def setUp(self):
        self.raw: str = '(S (NP-SBJ (PRP I)) (VP (VBP say) (NP (CD 1992))))'

    def test_parse(self):
        from grammar_env.corpus.action import parse
        parsed: list[str] = parse(self.raw)
        self.assertEqual(
            parsed,
            [
                '(', 'S', '(', 'NP-SBJ', '(', 'PRP', 'I', ')', ')',
                '(', 'VP', '(', 'VBP', 'say', ')', '(', 'NP', '(', 'CD', '1992', ')', ')', ')', ')'
            ])

    def test_get_actions(self):
        from grammar_env.corpus.action import get_actions, NT, Shift, Reduce, Action
        actions: list[Action] = get_actions(self.raw)
        self.assertEqual(
            actions, [
                NT('S'), NT('NP-SBJ'), Shift('PRP', 'I'), Reduce(),
                NT('VP'), Shift('VBP', 'say'), NT('NP'), Shift('CD', '1992'), Reduce(), Reduce(), Reduce()
            ])


class TestSentence(unittest.TestCase):
    def setUp(self):
        self.raw: str = '(S (NP-SBJ (PRP I)) (VP (VBP say) (NP (CD 1992))))'

    def test_sentence(self):
        from grammar_env.corpus.sentence import Sentence, GoldSpan
        from grammar_env.corpus.action import NT, Shift, Reduce
        sentence = Sentence(self.raw)

        self.assertEqual(sentence.raw, self.raw)
        self.assertEqual(sentence.symbols, ['i', 'say', '<num>'])
        self.assertEqual(sentence.actions, [
            NT('S'), NT('NP-SBJ'), Shift('PRP', 'I'), Reduce(),
            NT('VP'), Shift('VBP', 'say'), NT('NP'), Shift('CD', '1992'), Reduce(), Reduce(), Reduce()
        ])
        self.assertEqual(
            sentence.gold_spans_all,
            sorted([
                GoldSpan('NP-SBJ', 0, 0), GoldSpan('S', 0, 2), GoldSpan('VP', 1, 2), GoldSpan('NP', 2, 2)
            ]))
        self.assertEqual(
            sentence.gold_spans,
            sorted([
                GoldSpan('S', 0, 2), GoldSpan('VP', 1, 2)
            ]))
        self.assertEqual(
            sentence.tree_sr,
            ['S', 'S', 'S', 'R', 'R']
        )
        self.assertEqual(
            sentence.tree_sr_spans,
            [(0, 2), (1, 2)]
        )
        self.assertEqual(
            sentence.tree(),
            "(i (say <num>))"
        )

        print(sentence)
        print(sentence.tree())


class TestCorpus(unittest.TestCase):
    def setUp(self):
        self.ptb_path: Path = Path('resources/ptb.txt')

    def test_corpus(self):
        from grammar_env.corpus.corpus import Corpus
        corpus = Corpus(str(self.ptb_path))

        self.assertEqual(len(corpus), 13)

    def test_corpus_write(self):
        from grammar_env.corpus.corpus import Corpus
        from grammar_env.corpus.file import write_corpus_to_file
        corpus = Corpus(str(self.ptb_path))
        write_corpus_to_file(corpus, self.ptb_path.with_suffix('.json'))
        self.ptb_path.with_suffix('.json').unlink()

    def test_length(self):
        from grammar_env.corpus.corpus import Corpus
        from collections import defaultdict
        path: Path = Path('../../penn_treebank/ptb-train.txt')
        corpus = Corpus(str(path))

        sentence_lens: defaultdict[int, int] = defaultdict(int)
        for sentence in corpus.sentences:
            sentence_lens[len(sentence)] += 1

        print(sorted(sentence_lens.items()))
        print(len(corpus))


if __name__ == '__main__':
    unittest.main()
