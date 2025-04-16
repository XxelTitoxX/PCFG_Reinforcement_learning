import re
from dataclasses import dataclass, field
from functools import total_ordering
from logging import getLogger

from dataclasses_json import dataclass_json

from grammar_env.corpus.action import Action, NT, Reduce, Shift, get_actions

__all__ = ['GoldSpan', 'Sentence']

logger = getLogger(__name__)

POS_CLUSTER: dict[str, str] = {
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

POS_CLUSTER_TO_IDX: dict[str, int] = {
            "CC": 0,
            "CD": 1,
            "DT": 2,
            "IN": 3,
            "JJ": 4,
            "NN": 5,
            "PRP": 6,
            "RB": 7,
            "TO": 8,
            "VB": 9,
            "WH-": 10,
            "MD": 11,
            "POS": 12,
            "ETC": 13
        }

@dataclass_json
@dataclass(frozen=True)
@total_ordering
class GoldSpan:
    tag: str
    start: int
    end: int

    def __repr__(self):
        return f'GoldSpan({self.tag}, {self.start}, {self.end})'

    def __len__(self):
        return self.end - self.start + 1

    def __lt__(self, other):
        return (self.start, self.end, self.tag) < (other.start, other.end, other.tag)


def sanitize(symbol: str) -> str:
    """
    Replace numbers in raw with '<num>'.
    Lower case the symbol.
    For example, given 'I', return 'i'.
    Given '1992', return '<num>'.
    """
    return re.sub('[0-9]+([,.]?[0-9]*)*', '<num>', symbol).lower()


@dataclass_json
@dataclass
class Sentence:
    raw: str
    """
    Raw penn treebank tree.
    e.g., (S (NP-SBJ (PRP I)) (VP (VBP say) (NP (CD 1992))))
    """
    symbols: list[str] = field(init=False)
    """
    Symbols in the sentence.
    e.g., ['I', 'say', '<num>']
    """
    actions: list[Action] = field(init=False)
    """
    Actions in the sentence.
    e.g., [NT('S'), NT('NP-SBJ'), S('PRP', 'I'), R(), 
            NT('VP'), S('VBP', 'say'), NT('NP'), S('CD', '1992'), R(), R(), R()]
    """
    actions_sanitized: list[Action] = field(init=False)
    """
    Actions in the sentence with numbers replaced with '<num>', and symbols lower cased.
    e.g., [NT('S'), NT('NP-SBJ'), S('PRP', 'i'), R(),
            NT('VP'), S('VBP', 'say'), NT('NP'), S('CD', '<num>'), R(), R(), R()]
    """
    gold_spans_all: list[GoldSpan] = field(init=False)
    """
    List of gold spans in the sentence.
    e.g., [GoldSpan('NP-SBJ', 0, 0), GoldSpan('S', 0, 2), GoldSpan('VP', 1, 2), GoldSpan('NP', 2, 2)]
    """
    gold_spans: list[GoldSpan] = field(init=False)
    """
    List of gold spans in the sentence whose length is greater than 1.
    e.g., [GoldSpan('S', 0, 2), GoldSpan('VP', 1, 2)]

    We use gold_spans to calculate the f1 score.
    When calculating the f1 score, we ignore the whole sentence span.
    """
    tree_sr: list[str] = field(init=False)
    """
    Tree structure in Shift-Reduce format.
    e.g., ['S', 'S', 'S', 'R', 'R'], (I (say 1992))

    Note that all reduce actions are assumed to be binary.
    This does not hold for some penn treebank trees.
    i.e., reduce actions can reduce only one symbol or reduce more than two symbols.
    In such cases, (e.g., ((word1) word2 (word3))), tree_sr is (word1 (word2 word3)).

    Spans from tree_sr are used to calculate the oracle f1 score (the highest f1 score achievable).
    """
    tree_sr_spans: list[tuple[int, int]] = field(init=False)
    """
    List of spans in tree_sr.
    e.g., [(0, 2), (1, 2)]
    
    This is used to calculate the oracle f1 score.
    """
    pos_tags: list[int] = field(init=False)

    def __post_init__(self):
        self.actions: list[Action] = get_actions(self.raw)
        self.symbols: list[str] = []
        self.actions_sanitized: list[Action] = []

        for action in self.actions:
            match action:
                case Shift(tag, symbol):
                    self.symbols.append(sanitize(symbol))
                    self.actions_sanitized.append(Shift(tag, sanitize(symbol)))
                case _:
                    self.actions_sanitized.append(action)

        self._initialize_gold_and_sr()

        self.__pos_tags()

        num_shifts: int = len([action for action in self.tree_sr if action == 'S'])
        num_reduces: int = len([action for action in self.tree_sr if action == 'R'])
        assert num_shifts == len(self.symbols), (
            f"Number of shifts should be {len(self.symbols)}, but {num_shifts}"
        )
        assert num_reduces == len(self.symbols) - 1, (
            f"Number of reduces should be {len(self.symbols) - 1}, but {num_reduces}"
        )

    def _initialize_gold_and_sr(self) -> None:
        self.gold_spans_all: list[GoldSpan] = []
        self.gold_spans: list[GoldSpan] = []
        self.tree_sr: list[str] = []
        self.tree_sr_spans: list[tuple[int, int]] = []

        pointer: int = 0
        stack: list[tuple[int, int] | NT] = []
        """
        Given actions [
        NT('S'), NT('NP-SBJ'), S('PRP', 'I'), R(), 
        NT('VP'), S('VBP', 'say'), NT('NP'), S('CD', '1992'), R(), R(), R()
        ], algorithm is as follows:
        
        1. stack: [NT, NT, (0, 0)], gold_span: []
        2. stack: [NT, (0, 0)], gold_span: [(0, 0)]
        3. stack: [NT, (0, 0), NT, (1, 1), NT, (2, 2)], gold_span: [(0, 0)]
        4. stack: [NT (0, 0) NT (1, 1) (2, 2)], gold_span: [(0, 0), (2, 2)]
        5. stack: [NT (0, 0) (1, 2)], gold_span: [(0, 0), (2, 2), (1, 2)]
        6. stack: [(0, 2)], gold_span: [(0, 0), (2, 2), (1, 2), (0, 2)]
        """

        for action in self.actions:
            match action:
                case NT(tag):
                    stack.append(NT(tag))
                case Shift(_):
                    stack.append((pointer, pointer))
                    pointer += 1
                    self.tree_sr.append('S')
                case Reduce():
                    assert len(stack) > 0 and isinstance(stack[-1], tuple), f"Stack: {stack}"
                    span: tuple[int, int] = stack.pop()

                    n_new_shifts: int = 0
                    while isinstance(stack[-1], tuple):
                        n_new_shifts += 1
                        span = (stack.pop()[0], span[1])

                    assert len(stack) > 0 and isinstance(stack[-1], NT), f"Stack: {stack}"
                    nt: NT = stack.pop()
                    self.gold_spans_all.append(GoldSpan(nt.tag, span[0], span[1]))
                    stack.append(span)

                    # If there is an unary reduction, there can be multiple spans with the same start and end.
                    #
                    # Appending if n_new_shifts > 0,
                    # gold_spans won't have multiple spans with the same start and end.
                    # (We ignore the duplicate.)
                    #
                    # Appending if span[0] != span[1],
                    # gold_spans will have multiple spans with the same start and end.
                    # (We do not ignore the duplicate.)
                    #
                    # if n_new_shifts > 0:
                    #     self.gold_spans.append(GoldSpan(nt.tag, span[0], span[1]))
                    if span[0] != span[1]:
                        self.gold_spans.append(GoldSpan(nt.tag, span[0], span[1]))

                    left: int = span[0]
                    while n_new_shifts > 0:
                        self.tree_sr.append('R')
                        self.tree_sr_spans.append((left, span[1]))
                        n_new_shifts -= 1
                        left += 1

        assert len(stack) == 1 and isinstance(stack[0], tuple), f"Stack: {stack}"
        assert pointer == len(self.symbols), f"Pointer: {pointer}, Symbols: {self.symbols}"

        self.gold_spans_all.sort()
        self.gold_spans.sort()
        self.tree_sr_spans.sort()

    def tree(self) -> str:
        """
        From tree_sr, construct a tree structure.
        Assume symbols = [A, B, C, D], tree_sr = [S, S, R, S, S, R, R]
        S denotes Shift, R denotes Reduce.
        Then, the tree structure is:
        ((A B) (C D))

        :return: Tree structure in string format.
        """
        stack: list[str] = []
        symbols: list[str] = list(reversed(self.symbols))

        for action in self.tree_sr:
            assert action in ['S', 'R']

            if action == 'S':
                stack.append(symbols.pop())
            else:
                right = stack.pop()
                left = stack.pop()
                stack.append(f"({left} {right})")

        assert len(stack) == 1, f"Stack should have only one element, but {stack}"
        assert len(symbols) == 0, f"Symbols should be empty, but {symbols}"
        return stack[0]

    def __len__(self):
        return len(self.symbols)
    
    def __pos_tags(self):
        sentence_pos_tags = []
        for action in self.actions_sanitized:
            match action:
                case Shift(pos_tag, symbol):
                    sentence_pos_tags.append(POS_CLUSTER_TO_IDX[POS_CLUSTER[pos_tag]])
                case _:
                    pass
            self.pos_tags = sentence_pos_tags
