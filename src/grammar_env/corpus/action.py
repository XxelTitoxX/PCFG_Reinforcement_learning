from abc import ABC
from dataclasses import dataclass

from dataclasses_json import dataclass_json

__all__ = ['Action', 'Shift', 'Reduce', 'NT', 'get_actions']


@dataclass_json
@dataclass(frozen=True)
class Action(ABC):
    pass


@dataclass_json
@dataclass(frozen=True)
class Shift(Action):
    """Shift action."""
    tag: str
    symbol: str

    def __repr__(self):
        return f'S({self.tag}, {self.symbol})'


@dataclass_json
@dataclass(frozen=True)
class Reduce(Action):
    """Reduce action."""
    pass


@dataclass_json
@dataclass(frozen=True)
class NT(Action):
    """Non-terminal action."""
    tag: str

    def __repr__(self):
        return f'NT({self.tag})'


def get_word(raw: str, start_idx: int) -> str:
    word: str = ''
    for i in range(start_idx, len(raw)):
        assert raw[i] != '('
        if raw[i] == ' ' or raw[i] == ')':
            break
        word += raw[i]
    return word


def parse(raw: str) -> list[str]:
    """
    Parse raw penn treebank tree.

    :param raw: str
        e.g., (S (NP-SBJ (PRP I)) (VP (VBP say) (NP (CD 1992))))
    :return: list of strings
        e.g., ['(', 'S', '(', 'NP-SBJ', '(', 'PRP', 'I', ')', ')',
                '(', 'VP', '(', 'VBP', 'say', ')', '(', 'NP', '(', 'CD', '1992', ')', ')', ')', ')']
    """
    raw: str = raw.strip()
    parsed: list[str] = []

    i, max_idx = 0, len(raw)
    while i < max_idx:
        if raw[i] == '(' or raw[i] == ')':
            parsed.append(raw[i])
            i += 1

        elif raw[i] == ' ':
            i += 1

        else:
            word: str = get_word(raw, i)
            parsed.append(word)
            i += len(word)

    return parsed


def get_actions(raw: str) -> list[Action]:
    """
    Get actions from raw penn treebank tree.
    Note that NT and Reduce actions make a pair of parenthesis.
    
    :param raw: list of strings
        e.g., (S (NP-SBJ (PRP I)) (VP (VBP say) (NP (CD 1992))))
    :return: list of actions
        e.g., [NT('S'), NT('NP-SBJ'), Shift('PRP', 'I'), Reduce(), NT('VP'), 
                Shift('VBP', 'say'), NT('NP'), Shift('CD', '1992') Reduce(), Reduce(), Reduce()]
    """

    parsed: list[str] = parse(raw)
    actions: list[Action] = []

    i, max_idx = 0, len(parsed)
    while i < max_idx:
        assert parsed[i] == '(' or parsed[i] == ')', (
            f"parsed[{i}] must be '(' or ')', got {parsed[i]} in {parsed}"
        )
        if parsed[i] == '(':
            # Non-terminal or shift action
            # (tag (... or (tag symbol)
            i += 1
            tag: str = parsed[i]
            i += 1
            if parsed[i] == '(':
                # Non-terminal action
                actions.append(NT(tag))
            else:
                # Shift action
                symbol: str = parsed[i]
                i += 1
                assert parsed[i] == ')'
                i += 1
                actions.append(Shift(tag, symbol))
        else:
            # Reduce action
            actions.append(Reduce())
            i += 1

    num_nt: int = len([action for action in actions if isinstance(action, NT)])
    num_reduce: int = len([action for action in actions if isinstance(action, Reduce)])
    assert num_nt == num_reduce, (
        f"num_nt ({num_nt}) must be equal to num_reduce ({num_reduce}) in actions"
    )

    return actions
