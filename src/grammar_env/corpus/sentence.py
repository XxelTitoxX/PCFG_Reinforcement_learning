import re
from dataclasses import dataclass, field
from functools import total_ordering
from logging import getLogger

from dataclasses_json import dataclass_json

from grammar_env.corpus.action import Action, NT, Reduce, Shift, get_actions

__all__ = ['GoldSpan', 'Sentence']

logger = getLogger(__name__)

#DATASET = 'ptb'
DATASET = 'toy_grammar'

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

NT_TAG_TO_IDX = {
    'X': 0, 'ADJP': 1, 'ADVP': 2, 'CONJP': 3, 'FRAG': 4, 'INTJ': 5, 'LST': 6, 'NAC': 7, 'NP': 8,
    'NX': 9, 'PP': 10, 'PRN': 11, 'PRT': 12, 'QP': 13, 'RRC': 14, 'S': 15, 'SBAR': 16, 'SBARQ': 17,
    'SINV': 18, 'SQ': 19, 'UCP': 20, 'VP': 21, 'WHADJP': 22, 'WHADVP': 23, 'WHNP': 24, 'WHPP': 25
}

def toy_grammar_nt_tag_to_idx(tag: str) -> int:
    assert tag[:2] == 'NT', f"NT tag should start with 'NT', but got {tag}"
    return int(tag[2:])

def toy_grammar_pos_tag_to_idx(tag: str) -> int:
    assert tag[:2] == 'PT', f"PT tag should start with 'PT', but got {tag}"
    return int(tag[2:])

def ptb_nt_tag_to_idx(tag: str) -> int:
    return NT_TAG_TO_IDX[remove_after_dash(tag)]

def ptb_pos_tag_to_idx(tag: str) -> int:
    return POS_CLUSTER_TO_IDX[POS_CLUSTER[tag]]

def nt_tag_to_idx(tag: str) -> int:
    if DATASET == 'toy_grammar':
        return toy_grammar_nt_tag_to_idx(tag)
    elif DATASET == 'ptb':
        return ptb_nt_tag_to_idx(tag)
    else:
        raise ValueError(f"Unknown dataset: {DATASET}. Supported datasets are 'toy_grammar' and 'ptb'.")
    
def pos_tag_to_idx(tag: str) -> int:
    if DATASET == 'toy_grammar':
        return toy_grammar_pos_tag_to_idx(tag)
    elif DATASET == 'ptb':
        return ptb_pos_tag_to_idx(tag)
    else:
        raise ValueError(f"Unknown dataset: {DATASET}. Supported datasets are 'toy_grammar' and 'ptb'.")

def remove_after_dash(s:str) -> str:
    """
    Removes everything after the first '-' or '=' or '|' character in a string, including the character itself.
    
    Parameters:
        s (str): The input string.
    
    Returns:
        str: The truncated string.
    """
    return re.split('[-=|]', s, 1)[0]

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
    
    def start_end(self):
        return (self.start, self.end)


def ptb_sanitize(symbol: str) -> str:
    """
    Replace numbers in raw with '<num>'.
    Lower case the symbol.
    For example, given 'I', return 'i'.
    Given '1992', return '<num>'.
    """
    return re.sub('[0-9]+([,.]?[0-9]*)*', '<num>', symbol).lower()

def sanitize(symbol: str) -> str:
    if DATASET == 'ptb':
        return ptb_sanitize(symbol)
    return symbol


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
    symbols_idx: list[int] = field(init=False)
    """
    Indices of the words in the sentence, provided by global corpus indexing.
    e.g., [0, 1, 2] where 0 is 'I', 1 is 'say', and 2 is '<num>'.
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
    tree_sr_spans: list[GoldSpan] = field(init=False)
    """
    List of spans in tree_sr.
    e.g., [GoldSpan(gt_tag, 0, 2), GoldSpan(deduced_gt_tag, 1, 2)]
    
    This is used to calculate the oracle f1 score.
    """
    binary_gt_spans: dict[tuple[int, int], int] = field(init=False)
    """
    Dictionary of ground truth spans from binarised tree.
    Keys are tuples of (start, end) and values are the indexed ground truth tags.
    e.g., {(0, 2): 'S'=8, (1, 2): 'VP'=12}
    This is used to calculate the oracle f1 score.
    """
    pos_tags: list[int] = field(init=False)


    def __post_init__(self):
        self.actions: list[Action] = get_actions(self.raw)
        self.symbols: list[str] = []
        self.actions_sanitized: list[Action] = []

        for action in self.actions:
            if isinstance(action, Shift):
                self.symbols.append(sanitize(action.symbol))
                self.actions_sanitized.append(Shift(action.tag, sanitize(action.symbol)))
            else:
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
        self.binary_gt_spans: dict[tuple[int, int], int] = {}

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
            if isinstance(action, NT):
                stack.append(action)
            if isinstance(action, Shift):
                stack.append((pointer, pointer))
                pointer += 1
                self.tree_sr.append('S')
            if isinstance(action, Reduce):
                assert len(stack) > 0 and isinstance(stack[-1], tuple), f"Stack: {stack}"
                span: tuple[int, int] = stack.pop()

                intermediate_left: list[int] = []
                while isinstance(stack[-1], tuple):
                    span = (stack.pop()[0], span[1])
                    intermediate_left.append(span[0])

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

                for idx, left in enumerate(intermediate_left):
                    self.tree_sr_spans.append(GoldSpan(nt.tag+"'", left, span[1]))
                    self.tree_sr.append('R')
                    if idx == len(intermediate_left) - 1: # This is the root of the binarization : we give the original NT tag
                        self.binary_gt_spans[(left, span[1])] = nt_tag_to_idx(nt.tag) #*2
                    else: # This is an intermediate node of the binarization : we give the modified NT tag
                        self.binary_gt_spans[(left, span[1])] = nt_tag_to_idx(nt.tag) #*2 +1

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
            if isinstance(action, Shift):
                sentence_pos_tags.append(pos_tag_to_idx(action.tag))
            self.pos_tags = sentence_pos_tags
