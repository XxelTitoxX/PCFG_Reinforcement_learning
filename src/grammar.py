import numpy as np


def find_consecutive_pair(arr : np.array, a : np.array, b : np.array):
    batch_size = arr.shape[0]
    # Create a boolean array where elements are True where arr[k,i] == a[k] and arr[k,i+1] == b[k]
    condition = (arr[:,:-1] == a[:,None]) & (arr[:,1:] == b[:,None])
    
    # Get the indices where the condition is True
    idx = np.where(condition)

    full_idx = np.full(batch_size, -1)

    full_idx[idx[0]] = idx[1]  # Fill the indices for the rows where (a,b) was found
    return full_idx



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
    def __init__(self):
        pass

    def parsing_step(self, sentence : np.array, rule_idx:np.array):
        rule = self.bin_rules[rule_idx]
        lhs = rule[:,0]
        rhs1 = rule[:,1]
        rhs2 = rule[:,2]
        # Find the first occurrence of rhs1 and rhs2 in the sentence
        rhs_idx = find_consecutive_pair(sentence, rhs1, rhs2)

        valid_idx = np.where(rhs_idx != -1)[0]
        sentence[valid_idx, rhs_idx[valid_idx]:rhs_idx[valid_idx]+2] = lhs[valid_idx]
        valid_parsing = np.full(sentence.shape[0], False)
        valid_parsing[valid_idx] = True
        return sentence, valid_parsing

class CompleteBinGrammar(BinGrammar):
    def __init__(self, num_nt : int, num_pt : int):
        self.num_nt = num_nt
        self.num_pt = num_pt
        self.bin_rules : np.array = self.__build_bin_rules()

    def __build_bin_rules(self):
        self.bin_rules = np.indices((self.num_nt, self.num_nt+self.num_pt, self.num_nt+self.num_pt)).reshape(3, -1).T


    

class HierarchicalBinGrammar(BinGrammar) :
    def __init__(self, hierarchy_sizes : np.array, num_pt : int):
        assert hierarchy_sizes[0] == 1, "First hierarchy class must only contain start symbol"
        self.hierarchy_sizes = hierarchy_sizes
        self.num_products = (np.cumsum(hierarchy_sizes))[::-1]+num_pt
        self.num_pt = num_pt
        self.num_symbols = np.sum(hierarchy_sizes) + num_pt
        self.bin_rules : np.array = self.__build_bin_rules()

    def __build_bin_rules(self):
        bin_rules = []
        for i in range(self.num_products):
            class_rules = np.indices((self.hierarchy_sizes[i], self.num_products[i], self.num_products[i])).reshape(3, -1).T
            class_rules[:,0] += np.sum(self.hierarchy_sizes[:i])
            class_rules[:,1] += np.sum(self.hierarchy_sizes[:i])
            class_rules[:,2] += np.sum(self.hierarchy_sizes[:i])
            bin_rules.append(class_rules)
        self.bin_rules = np.concatenate(bin_rules, axis=0)



    
        