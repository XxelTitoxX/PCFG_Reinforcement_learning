import numpy as np
import torch
from typing import Optional

'''
Non-terminal with index 0 is the start symbol.
'''


def find_consecutive_pair(arr : torch.Tensor, a : torch.Tensor, b : torch.Tensor):
    batch_size = arr.shape[0]
    device = arr.device
    # Create a boolean array where elements are True where arr[k,i] == a[k] and arr[k,i+1] == b[k]
    condition = (arr[:,:-1] == a[:,None]) & (arr[:,1:] == b[:,None])
    
    # Get the indices where the condition is True
    idx = torch.where(condition)

    full_idx = torch.full((batch_size,), -1, dtype=torch.int32, device=device)
    full_idx[idx[0]] = idx[1].to(dtype=torch.int32)  # Fill the indices for the rows where (a,b) was found
    
    return full_idx

def fuse_and_shift(arr: torch.Tensor, idx: torch.Tensor, value:torch.Tensor, padding_value):
    assert torch.all((idx >= 0) & (idx < arr.shape[1] - 1)), "Index out of bounds"
    
    arr_copy = arr.clone()
    value = value.to(arr_copy.dtype)
    # Insert fusion value
    arr_copy[:,idx] = value
    
    # Shift elements from i+2 to end
    for i in range(arr.shape[0]):
        # Shift elements in the row
        arr_copy[i, idx[i]+1:-1] = arr[i, idx[i]+2:]
    #arr_np[:,idx_np+1:-1] = arr_np[:,idx_np+2:]
    
    # Pad the last element
    arr_copy[:, -1] = padding_value

    return arr_copy

def fuse_spans_and_shift(arr: torch.Tensor, idx: torch.Tensor, padding_value):
    assert torch.all((idx >= 0) & (idx < arr.shape[1] - 1)), "Index out of bounds"
    assert idx.shape[0] == arr.shape[0], "Index and array must have the same number of rows"
    
    arr_copy = arr.clone()

    # Fuse spans
    fused_spans = torch.stack([arr_copy[torch.arange(arr.shape[0]), idx, 0], arr_copy[torch.arange(arr.shape[0]), idx + 1, 1]], dim=1)
    arr_copy[:,idx, :] = fused_spans
    
    # Shift elements from i+2 to end
    for i in range(arr.shape[0]):
        # Shift elements in the row
        arr_copy[i, idx[i]+1:-1] = arr[i, idx[i]+2:]
    
    # Pad the last element
    arr_copy[:, -1, :] = padding_value

    return arr_copy, fused_spans



from collections import defaultdict




class SupervisedUnaryGrammar:
    def __init__(self, corpus):
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

        rules: np.array = np.zeros(num_pt, num_t)
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
        self.rules: np.array = rules
        self.num_t: int = num_t
        self.num_pt: int = num_pt

    def _get_rule_log_probs(self, sentences: np.array) -> np.array:
        batch_size, max_sentence_length = sentences.shape
        num_pt: int = self.num_pt

        # Expand self.rules to shape (batch_size, num_pt, num_t)
        rules_expanded = self.rules.unsqueeze(0).expand(batch_size, num_pt, self.num_t)
        # Expand sentences to shape (batch_size, num_pt, max_sentence_length)
        sentences_expanded = sentences.unsqueeze(1).expand(batch_size, num_pt, max_sentence_length)

        # Gather the rule log probabilities for each terminal in the sentences
        rule_log_probs = np.take_along_axis(rules_expanded, sentences_expanded, axis=2)
        assert rule_log_probs.shape == (batch_size, num_pt, max_sentence_length)

        return rule_log_probs



class BinGrammar:
    def __init__(self, device: torch.device):
        self.bin_rules: Optional[torch.Tensor] = None
        self.num_nt: Optional[int] = None
        self.num_pt: Optional[int] = None
        self.device: torch.device = device

    def parsing_step(self, sentence: torch.Tensor, rule_idx: torch.Tensor):
        rule = self.bin_rules[rule_idx]
        lhs = rule[:, 0]
        rhs1 = rule[:, 1]
        rhs2 = rule[:, 2]
        # Find the first occurrence of rhs1 and rhs2 in the sentence
        rhs_idx = find_consecutive_pair(sentence, rhs1, rhs2)
        valid_idx = torch.where(rhs_idx != -1)[0]
        sentence[valid_idx] = fuse_and_shift(sentence[valid_idx], rhs_idx[valid_idx], lhs[valid_idx], -1)
        valid_parsing = torch.full((sentence.shape[0],), False, device=self.device)
        valid_parsing[valid_idx] = True
        return sentence, valid_parsing, rhs_idx

    def span_computation_step(self, rhs_index: torch.Tensor, spans_sentences: torch.Tensor):
        valid_idx = torch.where(rhs_index != -1)[0]
        if valid_idx.shape[0] == 0:
            return spans_sentences, torch.full((spans_sentences.shape[0], 2), -1, dtype=torch.int32, device=self.device)
        # Find the first occurrence of rhs1 and rhs2 in the sentence
        next_spans_sentences, valid_new_spans = fuse_spans_and_shift(spans_sentences[valid_idx], rhs_index[valid_idx], -1)
        next_spans = spans_sentences.clone()
        next_spans[valid_idx] = next_spans_sentences
        new_spans = torch.full((next_spans.shape[0], 2), -1, dtype=torch.int32, device=self.device)
        new_spans[valid_idx] = valid_new_spans
        return next_spans, new_spans

    def num_rules(self):
        return self.bin_rules.shape[0]


class CompleteBinGrammar(BinGrammar):
    def __init__(self, num_nt: int, num_pt: int, device: torch.device):
        super().__init__(device=device)
        assert num_nt > 0 and num_pt > 0, "Number of non-terminals and pre-terminals must be greater than 0"
        self.num_nt = num_nt
        self.num_pt = num_pt
        self.__build_bin_rules()

    def __build_bin_rules(self):
        # Equivalent to np.indices but using torch
        idx0, idx1, idx2 = torch.meshgrid(
            torch.arange(self.num_nt, device=self.device),
            torch.arange(self.num_nt + self.num_pt - 1, device=self.device),
            torch.arange(self.num_nt + self.num_pt - 1, device=self.device),
            indexing='ij'
        )
        indices = torch.stack([idx0.flatten(), idx1.flatten(), idx2.flatten()], dim=1)
        self.bin_rules = indices
        self.bin_rules[:, 1:] += 1


class HierarchicalBinGrammar(BinGrammar):
    def __init__(self, hierarchy_sizes: torch.Tensor, num_pt: int, device: torch.device):
        assert hierarchy_sizes[0] == 1, "First hierarchy class must only contain start symbol"
        super().__init__(device=device)
        
        # Convert hierarchy_sizes to torch tensor if it's not already
        if not isinstance(hierarchy_sizes, torch.Tensor):
            hierarchy_sizes = torch.tensor(hierarchy_sizes, device=device)
        
        self.hierarchy_sizes = hierarchy_sizes
        self.num_products = (torch.cumsum(hierarchy_sizes, dim=0).flip(dims=[0])) + num_pt
        self.num_pt = num_pt
        self.num_nt = torch.sum(hierarchy_sizes).item()
        self.num_symbols = self.num_nt + self.num_pt
        self.__build_bin_rules()

    def __build_bin_rules(self):
        bin_rules = []
        for i in range(len(self.num_products)):
            # Create indices
            idx0, idx1, idx2 = torch.meshgrid(
                torch.arange(self.hierarchy_sizes[i], device=self.device),
                torch.arange(self.num_products[i], device=self.device),
                torch.arange(self.num_products[i], device=self.device),
                indexing='ij'
            )
            class_rules = torch.stack([idx0.flatten(), idx1.flatten(), idx2.flatten()], dim=1)
            
            # Adjust indices
            class_rules[:, 0] += torch.sum(self.hierarchy_sizes[:i]).item()
            class_rules[:, 1] += torch.sum(self.hierarchy_sizes[:i]).item()
            class_rules[:, 2] += torch.sum(self.hierarchy_sizes[:i]).item()
            
            bin_rules.append(class_rules)
        
        self.bin_rules = torch.cat(bin_rules, dim=0)



    
        