import numpy as np
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
import argparse
import os
from pathlib import Path
import json
from nltk import CFG as NLTK_CFG, Nonterminal, Production
from nltk.parse.generate import generate

rng = np.random.default_rng(seed=0)  # Use the default random number generator from NumPy
np.random.seed(0)  # Set a random seed for reproducibility

MAX_TRIES = 100  # Maximum number of tries to generate n_sentences unique strings
EARLY_STOP = 10  # Early stop if number of unique strings does not increase for this many tries

def random_partition(V:float, K:int) -> np.ndarray:
    values = np.random.dirichlet(np.ones(K)) * V  # Dirichlet distribution to generate random partition
    return values

def generate_sentence_worker(cfg_params:dict, max_length:int, n_sentences:int, seed:int) -> list:
    np.random.seed(seed)
    cfg = CFG(**cfg_params)
    return [cfg._generate_sentence(max_length) for _ in range(n_sentences)]

def to_nltk_cfg(grammar: 'CFG') -> NLTK_CFG:
    """
    Converts this CFG object to an nltk.CFG object.
    """

    productions = []

    # Map integer symbols to string labels
    def symbol(idx):
        if idx < grammar.n_non_terminals:
            return f'NT{idx}'
        elif idx < grammar.n_non_terminals + grammar.n_pre_terminals:
            return f'PT{idx - grammar.n_non_terminals}'
        else:
            return f'TT{idx - grammar.n_non_terminals - grammar.n_pre_terminals}'

    # Add n-ary rules
    for rule in grammar.nary_rules:
        lhs_idx = rule[0]
        rhs_indices = rule[1:]
        rhs_indices = rhs_indices[rhs_indices != -1]  # Remove padding

        lhs_sym = Nonterminal(symbol(lhs_idx))
        rhs_syms = [Nonterminal(symbol(idx)) for idx in rhs_indices]

        productions.append(Production(lhs_sym, rhs_syms))

    # Add unary rules (pre-terminal to terminal)
    for pt_idx, t_idx in grammar.unary_rules:
        lhs_sym = Nonterminal(symbol(pt_idx))
        rhs_sym = symbol(t_idx)
        productions.append(Production(lhs_sym, [rhs_sym]))

    # Define start symbol
    start_sym = Nonterminal(f'NT{grammar.start_symbol}')

    return NLTK_CFG(start=start_sym, productions=productions)

def nltk_cfg_to_string(cfg: NLTK_CFG) -> str:
    """
    Converts an nltk.CFG object to a string representation.
    """
    return '\n'.join(str(prod) for prod in cfg.productions())



class TreeNode:
    def __init__(self, type:str, index: int, children: Optional[list] = None):
        self.type : str = type  # The type of the node: 'NT' for non-terminal, 'PT' for pre-terminal, 'TT' for terminal
        self.index : int = index 
        self.children : list[TreeNode] = children if children is not None else []  # The children of the node (if any)
    
    def __repr__(self):
        if not self.children:
            return f"{self.symbol()}"
        children_repr = ' '.join(repr(child) for child in self.children)
        return f"({self.symbol()} {children_repr})"
    def symbol(self) -> str:
        """
        Returns the symbol of the node as a string.
        """
        return f"{self.type}{self.index}"
    def words_sentence(self) -> str:
        """
        Returns the raw sequence of words represented by this tree node.
        """
        if self.type == 'TT':
            return f"{self.symbol()}"
        else:
            children_w_sentences = [child.words_sentence() for child in self.children]
            non_empty_children = [child for child in children_w_sentences if child]
            return ' '.join(non_empty_children)

class Sentence:
    def __init__(self, root: TreeNode, length: Optional[int] = None):
        self.root = root  # The root of the sentence tree
        self.length = length if length is not None else self._calculate_length(root)

    def __repr__(self):
        """
        Returns a string representation of the sentence (PTB format).
        """
        return self.root.__repr__()
    
    def words_sentence(self) -> str:
        """
        Returns the raw sequence of words represented by this sentence.
        """
        return self.root.words_sentence()
        
    
    def _calculate_length(self, node: TreeNode) -> int:        
        """
        Calculates the length of the sentence by counting the number of terminal nodes.
        """
        if not node.children:
            return 1
        return sum(self._calculate_length(child) for child in node.children)
    
def save_sentences(strings:list[str], save_dir:str) -> None:
        """
        Saves the generated sentences to a file in PTB format.
        """

        # Split into training and validation sets
        train_size = int(0.8 * len(strings))
        train_sentences = strings[:train_size]
        val_sentences = strings[train_size:]

        train_path = os.path.join(save_dir, 'train_sentences.txt')
        val_path = os.path.join(save_dir, 'val_sentences.txt')
        with open(train_path, 'w') as f:
            for sentence in train_sentences:
                f.write(sentence + '\n')
        with open(val_path, 'w') as f:
            for sentence in val_sentences:
                f.write(sentence + '\n')
        print(f"Saved {len(strings)} sentences to {save_dir}")


class CFG:
    def __init__(self, n_non_terminals:int, n_pre_terminals:int, n_terminals:int, n_nary_rules:int, max_pt_per_terminal:int=1, ambiguity_prob=0.0, max_unary_rules:Optional[int]=None, value_of_nt:int = 2, max_value_per_rhs:int = 7, unary_rules:Optional[np.ndarray]=None, nary_rules:Optional[np.ndarray]=None, min_nt_value:Optional[np.ndarray]=None, min_rule_value:Optional[np.ndarray]=None):
        assert n_non_terminals > 0 and n_pre_terminals > 0 and n_terminals > n_pre_terminals and n_nary_rules > 0 and max_pt_per_terminal > 0 and ambiguity_prob >= 0.0 and ambiguity_prob <= 1.0 and value_of_nt >= 1.0 and max_value_per_rhs > 0, \
            "Incorrect parameters"
        if max_unary_rules is not None:
            assert max_unary_rules >= n_terminals, "max_unary_rules must be greater than n_terminals"

        self.n_non_terminals :int = n_non_terminals
        self.n_pre_terminals :int = n_pre_terminals
        self.n_terminals :int = n_terminals
        self.n_nary_rules :int = n_nary_rules
        self.max_pt_per_terminal :int = max_pt_per_terminal
        self.ambiguity_prob :float = ambiguity_prob
        self.max_unary_rules :Optional[int] = max_unary_rules
        self.value_of_nt :int = value_of_nt
        self.max_value_per_rhs :int = max_value_per_rhs
        self.start_symbol :int = 0  # The start symbol of the grammar

        self.unary_rules :np.ndarray = unary_rules  # The unary rules of the grammar
        self.nary_rules :np.ndarray = nary_rules  # The n-ary rules of the grammar
        self.min_nt_value :np.ndarray = min_nt_value  # The minimum expansion length of each non-terminal
        self.min_rule_value :np.ndarray = min_rule_value  # The minimum expansion length of each rule

    def random_initialize(self):
        self.__gen_unary_rules()
        self.__gen_nary_rules()
        self.__compute_min_terminal_length()

    def retrieve_params(self) -> dict:
        """
        Returns the parameters of the CFG as a dictionary.
        """
        return {
            'n_non_terminals': self.n_non_terminals,
            'n_pre_terminals': self.n_pre_terminals,
            'n_terminals': self.n_terminals,
            'n_nary_rules': self.n_nary_rules,
            'max_pt_per_terminal': self.max_pt_per_terminal,
            'ambiguity_prob': self.ambiguity_prob,
            'max_unary_rules': self.max_unary_rules,
            'value_of_nt': self.value_of_nt,
            'max_value_per_rhs': self.max_value_per_rhs,
            'unary_rules': self.unary_rules,
            'nary_rules': self.nary_rules,
            'min_nt_value': self.min_nt_value,
            'min_rule_value': self.min_rule_value
        }



    def __gen_unary_rules(self):
        """
        Initializes the unary rules of the grammar as a 2D integer array where first column is pre-terminal and second is terminal.
        Each terminal can have multiple pre-terminal assignments, controlled by the `max_pt_per_terminal` and `ambiguity_prob` parameters.
        """
        terminals_idx = np.arange(self.n_terminals)
        """
        all_non_zero = True
        while all_non_zero:
            pt_dist = np.random.dirichlet(np.ones(self.n_pre_terminals)) # Dirichlet distribution
            all_non_zero = np.all(pt_dist > 0.0)  # Ensure all pre-terminals have a non-zero probability
        """
        
        pt_dist = np.ones(self.n_pre_terminals) / self.n_pre_terminals  # Uniform distribution
        one_t_per_pt = False
        while not one_t_per_pt:
            default_assignment = np.random.choice(self.n_pre_terminals, size=self.n_terminals, p=pt_dist)
            one_t_per_pt = np.all(np.bincount(default_assignment, minlength=self.n_pre_terminals) >= 1)
        self.unary_rules = np.stack((default_assignment, terminals_idx), axis=1)
        for _ in range(self.max_pt_per_terminal):
            t_bool_idx = np.random.binomial(1, self.ambiguity_prob, size=self.n_terminals).astype(bool)
            t_idx = terminals_idx[t_bool_idx]
            pt_choice = np.random.choice(self.n_pre_terminals, size=np.sum(t_bool_idx), p=pt_dist)
            new_unary_rules = np.stack((pt_choice, t_idx), axis=1)
            self.unary_rules = np.concatenate((self.unary_rules, new_unary_rules), axis=0)
        _, unique_idx = np.unique(self.unary_rules, axis=0, return_index=True)
        self.unary_rules = self.unary_rules[np.sort(unique_idx)]  # Remove duplicates
        self.unary_rules += self.n_non_terminals  # We shift the pre-terminals to the right by n_non_terminals to avoid overlap with non-terminals
        self.unary_rules[:, 1] += self.n_pre_terminals  # We shift the terminals to the right by n_pre_terminals to avoid overlap with pre-terminals
        if self.max_unary_rules is not None:
            self.unary_rules = self.unary_rules[:self.max_unary_rules, :] 



    def __gen_nary_rules(self):
        """
        Generates n-ary rules for the grammar. 
        Each rule consists in one non-terminal on the LHS and two or more non-terminals or pre-terminals on the RHS.
        non-terminals are indexed from 0 to n_non_terminals-1, pre-terminals from n_non_terminals to n_non_terminals+n_pre_terminals-1, and terminals from n_non_terminals+n_pre_terminals to n_non_terminals+n_pre_terminals+n_terminals-1.
        We handle rules with multiple lengths of RHS by using -1 as padding symbol.
        The first n_non_terminals rows of the nary_rules array are reserved for termination rules, which have only pre-terminals on the RHS.
        """
        # We first generate rules where the RHS is only pre-terminals to ensure termination of non-terminal expansion
        termination_rules = np.full((self.n_non_terminals, self.max_value_per_rhs+1), -1, dtype=int)
        termination_rules[:, 0] = np.arange(self.n_non_terminals)  # The LHS is the non-terminal itself
        termination_rules_values = np.random.randint(1, self.max_value_per_rhs + 1, size=self.n_non_terminals)
        for i in range(self.max_value_per_rhs):
            incomplete_rules_idx = np.where(termination_rules_values > i)[0]
            n_incomplete = len(incomplete_rules_idx)
            # We draw the pre-terminal uniformly at random
            pre_terminal = np.random.randint(self.n_non_terminals, self.n_non_terminals + self.n_pre_terminals, size=n_incomplete)
            # We assign the pre-terminal to the incomplete rules
            termination_rules[incomplete_rules_idx, i+1] = pre_terminal

        # We also generate one expansion rule for the start symbol
        rule_len = np.random.randint(2, max(int(self.max_value_per_rhs/self.value_of_nt), 2))
        start_symbol_rule = np.full((1, self.max_value_per_rhs+1), -1, dtype=int)
        expansion_rhs = np.random.randint(1, self.n_non_terminals, size=rule_len)
        start_symbol_rule[0, 0] = self.start_symbol  # The LHS is the start symbol
        start_symbol_rule[0, 1:rule_len+1] = expansion_rhs

        # If the number of termination rules exceeds the number of n-ary rules, we can stop here
        n_nary_rules = self.n_nary_rules - self.n_non_terminals - 1
        if n_nary_rules <= 0:
            self.nary_rules = np.concatenate((termination_rules, start_symbol_rule), axis=0)
            return
        
        # We now generate the n-ary rules
        nary_rules = np.full((n_nary_rules, self.max_value_per_rhs+1), -1, dtype=int)
        lhs = np.random.randint(0, self.n_non_terminals, size=n_nary_rules)
        nary_rules[:, 0] = lhs

        # We draw the value of the RHS uniformly at random, between 1 and max_value_per_rhs (we allow non-binary rules)
        rhs_values = np.random.randint(1, self.max_value_per_rhs + 1, size=n_nary_rules)
        current_rhs_values = np.zeros(n_nary_rules, dtype=np.float32)

        for i in range(self.max_value_per_rhs):
            # We find the incomplete rules, i.e. those that have not yet reached the maximum value of the RHS
            incomplete_rules = (rhs_values - current_rhs_values) >= 1
            # We select the incomplete rules with remaining value less than NT
            only_pt_mask = ((rhs_values - current_rhs_values) < self.value_of_nt) & incomplete_rules
            n_only_pt = only_pt_mask.sum()

            pt_or_nt_mask = ((rhs_values - current_rhs_values) >= self.value_of_nt) & incomplete_rules
            n_pt_or_nt = pt_or_nt_mask.sum()

            # We draw one pre-terminal for each incomplete rule with remaining value <= NT
            pt = np.random.randint(self.n_non_terminals, self.n_non_terminals + self.n_pre_terminals, size=n_only_pt)

            # We draw one pre-terminal or non-terminal for each incomplete rule with remaining value greater than NT
            pt_or_nt = np.random.randint(0, self.n_non_terminals + self.n_pre_terminals, size=n_pt_or_nt)

            nary_rules[only_pt_mask, i+1] = pt
            nary_rules[pt_or_nt_mask, i+1] = pt_or_nt

            # We update the current value of the RHS
            current_rhs_values[only_pt_mask] += 1
            current_rhs_values[pt_or_nt_mask] += 1

            nt_only = pt_or_nt >= self.n_non_terminals
            pt_or_nt_idx = np.arange(n_nary_rules)[pt_or_nt_mask]
            nt_idx = pt_or_nt_idx[nt_only]
            current_rhs_values[nt_idx] += (self.value_of_nt-1)

        # We concatenate the termination rules with the n-ary rules
        self.nary_rules = np.concatenate((termination_rules, start_symbol_rule, nary_rules), axis=0)

    def __compute_min_terminal_length(self) -> int:
        """
        Computes the minimum expansion length of any non-terminal in the grammar.
        This is the minimum number of terminal symbols that can be generated from a non-terminal.
        """
        # We start with the assumption that the minimum length of each NT is the shortest termination rule
        # NOTE: There is possibly more than one termination rule, so we take the minimum length of all of them
        min_lengths = np.full(self.n_non_terminals, np.inf)
        termination_rules_mask = np.all((self.nary_rules[:, 1:] == -1) | ((self.nary_rules[:, 1:] >= self.n_non_terminals) & (self.nary_rules[:, 1:] < (self.n_non_terminals + self.n_pre_terminals))), axis=1)
        termination_rules = self.nary_rules[termination_rules_mask]
        termination_rules_lengths = np.sum(termination_rules != -1, axis=1) - 1  # We subtract 1 because the first column is the LHS
        for i in range(self.n_non_terminals):
            if not np.any(termination_rules[:, 0] == i):
                raise ValueError(f"Non-terminal {i} has no termination rules.")
            min_lengths[i] = np.min(termination_rules_lengths[termination_rules[:, 0] == i])

        
        def current_length(symbol_idx: int) -> int:
            if symbol_idx < 0:
                return 0
            if symbol_idx >= self.n_non_terminals:
                return 1
            return min_lengths[symbol_idx]
        vectorized_curr_length = np.vectorize(current_length)

        # We now compute the minimum length of each NT considering all the n-ary rules
        saved_min_lengths = min_lengths.copy()
        while np.any(min_lengths != saved_min_lengths):
            saved_min_lengths = min_lengths.copy()
            for i in range(self.n_non_terminals):
                # We find all the n-ary rules that have this NT on the LHS
                nary_rules_mask = self.nary_rules[:, 0] == i
                nary_rules = self.nary_rules[nary_rules_mask]
                # We compute the length of the RHS of each rule
                rhs_lengths = np.sum(vectorized_curr_length(nary_rules[:, 1:]), axis=1)

                # We update the minimum length of this NT
                if len(rhs_lengths) > 0:
                    min_lengths[i] = min(min_lengths[i], np.min(rhs_lengths))

        self.min_nt_value = min_lengths
        self.min_rule_value = np.sum(vectorized_curr_length(self.nary_rules[:, 1:]), axis=1)

    def __get_min_terminal_length(self, symbol_idx:int) -> int:
        if symbol_idx < 0:
            return 0
        if symbol_idx >= self.n_non_terminals:
            return 1
        if self.min_nt_value is None:
            raise ValueError("Minimum terminal length has not been computed yet. Call __compute_min_terminal_length() first.")
        return self.min_nt_value[symbol_idx]


            

    def _generate_sentence(self, max_length:int) -> Sentence:
        if max_length < self.min_nt_value[self.start_symbol]:
            raise ValueError(f"Maximum length {max_length} is less than the minimum length of the start symbol NT{self.start_symbol}, which is {self.min_nt_value[self.start_symbol]}.")
        vectorized_min_length = np.vectorize(self.__get_min_terminal_length)
        def expand_node(node: TreeNode, budget: float):
            """
            Recursively expands a node until it reaches the budget or a terminal node.
            """
            if node.type == 'TT':
                return
            if node.type == 'PT':
                if budget < 1:
                    raise ValueError(f"Budget {budget} is less than 1, cannot expand pre-terminal node {node.index}.")
                global_index = node.index + self.n_non_terminals
                valid_unary_rules = self.unary_rules[self.unary_rules[:, 0] == global_index]
                n_valid_unary_rules = valid_unary_rules.shape[0]
                random_rule_idx = rng.integers(0, n_valid_unary_rules)
                terminal_idx = valid_unary_rules[random_rule_idx, 1]
                node.children = [TreeNode(type='TT', index=terminal_idx-self.n_pre_terminals-self.n_non_terminals)]  # Adjust index to match terminal range
                return
            if node.type == 'NT':
                if budget < self.min_nt_value[node.index]:
                    raise ValueError(f"Budget {budget} is less than the minimum length of non-terminal NT{node.index}, which is {self.min_nt_value[node.index]}.")
                # Global index is same as NT index
                valid_rules_mask = (self.nary_rules[:, 0] == node.index) & (self.min_rule_value <= budget)
                valid_nary_rules = self.nary_rules[valid_rules_mask]
                n_valid_nary_rules = valid_nary_rules.shape[0]
                random_rule_idx = rng.integers(0, n_valid_nary_rules)
                rule_rhs = valid_nary_rules[random_rule_idx, 1:]  # Get the RHS
                rule_rhs = rule_rhs[rule_rhs != -1]
                node.children = [TreeNode(type='NT', index=idx) if idx<self.n_non_terminals else TreeNode(type='PT', index=idx-self.n_non_terminals) for idx in rule_rhs]

                # Expand each child node recursively
                # We first need to define the budget for each child node
                exceeding_budget = budget - self.min_rule_value[valid_rules_mask][random_rule_idx]
                exceeding_budgets = random_partition(exceeding_budget, len(node.children))
                budgets = exceeding_budgets + vectorized_min_length(rule_rhs)
                for child, child_budget in zip(node.children, budgets):
                    expand_node(child, child_budget)
        # Start with the root node
        root = TreeNode(type='NT', index=self.start_symbol)
        # Expand the root node with the maximum length budget
        expand_node(root, max_length)
        return Sentence(root)
    
    def save_grammar(self, save_dir:str) -> None:
        """
        Saves the grammar parameters to a file in the specified directory.
        The parameters are saved as a dictionary in a text file.
        """
        params = self.retrieve_params()
        params_path = os.path.join(save_dir, 'grammar_params.json')
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                params[key] = value.tolist()
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Saved grammar parameters to {params_path}")

    def save_grammar_as_nltk(self, save_path:str) -> None:
        """
        Saves the grammar as an nltk.CFG object to a file in the specified directory.
        The CFG is saved in a text file format.
        """
        nltk_cfg = to_nltk_cfg(self)
        cfg_string = nltk_cfg_to_string(nltk_cfg)
        with open(save_path, 'w') as f:
            f.write(cfg_string)
        print(f"Saved grammar as nltk CFG to {save_path}")


    
    def generate_sentences(self, n_sentences:int, max_length:int) -> list[Sentence]:
        """
        Generates sentences from the grammar.
        Each sentence is a complete tree structure with start symbol 0 as root and terminals as leaves.
        """
        sentences = []
        n_workers = 4
        average_n_sentences = n_sentences // n_workers
        remaining_sentences = n_sentences - average_n_sentences * n_workers
        n_sentences_per_worker = [average_n_sentences] * n_workers
        n_sentences_per_worker[-1] += remaining_sentences  # Add the remaining sentences to the last worker
        cfg_params = self.retrieve_params()
        seeds = rng.integers(0, 1000000, size=n_workers)  # Generate random seeds for each worker

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(generate_sentence_worker, cfg_params, max_length, n_sentences_per_worker[i], seeds[i]) for i in range(n_workers)]
            for future in futures:
                sentences.extend(future.result())
        return list(sentences)
    
    def generate_unique_strings(self, n_sentences:int, max_length:int, format:str='ptb') -> list[str]:
        """
        Generates unique strings from the grammar.
        """
        set_strings = set()
        n_strings = 0
        n_tries = 0
        early_stop_counter = 0
        save_n_strings = 0
        try:
            while n_strings < n_sentences and n_tries < MAX_TRIES and early_stop_counter < EARLY_STOP:
                save_n_strings = n_strings
                sentences = self.generate_sentences(n_sentences, max_length)
                for sentence in sentences:
                    if format == 'ptb':
                        set_strings.add(str(sentence))
                    elif format == 'raw':
                        set_strings.add(sentence.words_sentence())
                    else:
                        raise ValueError(f"Unknown format: {format}. Supported formats are 'ptb' and 'raw'.")
                    n_strings = len(set_strings)
                    if n_strings >= n_sentences:
                        break
                if n_strings == save_n_strings:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
                n_tries += 1
                if n_tries % 1 == 0:
                    print(f"Attempt {n_tries}: Generated {n_strings} unique strings so far.")
            if n_strings < n_sentences:
                if n_tries >= MAX_TRIES:
                    print(f"Could not generate {n_sentences} unique strings within {MAX_TRIES} tries. Generated only {n_strings} unique strings.")
                if early_stop_counter >= EARLY_STOP:
                    print(f"Early stopping after {EARLY_STOP} attempts without increasing the number of unique strings. Generated only {n_strings} unique strings.")
        except KeyboardInterrupt:
            pass  # Allow the user to interrupt the generation process
        except Exception as e:
            print(f"An error occurred during string generation: {e}")
        return list(set_strings)
    
    def generate_strings_from_nltk(self, n_sentences:int, max_depth:int) -> list[Sentence]:
        """
        Generates sentences from the grammar using nltk's generate function.
        """
        nltk_cfg = to_nltk_cfg(self)
        sentences = set()
        for sentence in generate(nltk_cfg, depth=max_depth):
            sentences.add(' '.join(sentence))  # Join the words in the sentence
            if len(sentences) >= n_sentences:
                break
        if len(sentences) < n_sentences:
            print(f"Warning: Only {len(sentences)} unique sentences can be generated.")
        return list(sentences)
        
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentences from a CFG.")
    parser.add_argument('--save_dir', type=Path, required=True, help="Directory to save the generated grammar and sentences.")
    parser.add_argument('--n_non_terminals', type=int, required=True, help="Number of non-terminals in the grammar.")
    parser.add_argument('--n_pre_terminals', type=int, required=True, help="Number of pre-terminals in the grammar.")
    parser.add_argument('--n_terminals', type=int, required=True, help="Number of terminals in the grammar.")
    parser.add_argument('--n_nary_rules', type=int, required=True, help="Number of n-ary rules in the grammar.")
    parser.add_argument('--max_pt_per_terminal', type=int, default=1, help="Maximum number of pre-terminals per terminal.")
    parser.add_argument('--ambiguity_prob', type=float, default=0.0, help="Probability of ambiguity in the grammar.")
    parser.add_argument('--max_unary_rules', type=int, default=None, help="Maximum number of unary rules in the grammar.")
    parser.add_argument('--value_of_nt', type=float, default=1.5, help="Value of non-terminals in the grammar.")
    parser.add_argument('--max_value_per_rhs', type=int, default=7, help="Maximum value per right-hand side in the grammar.")
    parser.add_argument('--n_sentences', type=int, required=True, help="Number of sentences to generate.")
    parser.add_argument('--max_length', type=int, required=True, help="Maximum length of the sentences to generate.")
    args = parser.parse_args()
    cfg = CFG(
        n_non_terminals=args.n_non_terminals,
        n_pre_terminals=args.n_pre_terminals,
        n_terminals=args.n_terminals,
        n_nary_rules=args.n_nary_rules,
        max_pt_per_terminal=args.max_pt_per_terminal,
        ambiguity_prob=args.ambiguity_prob,
        max_unary_rules=args.max_unary_rules,
        value_of_nt=args.value_of_nt,
        max_value_per_rhs=args.max_value_per_rhs
    )
    cfg.random_initialize() 
    absolute_save_dir = args.save_dir.resolve()

    sentences = cfg.generate_unique_strings(args.n_sentences, args.max_length)
    if not os.path.exists(absolute_save_dir):
        os.makedirs(absolute_save_dir)
    save_sentences(sentences, absolute_save_dir)
    cfg.save_grammar(absolute_save_dir)



        





