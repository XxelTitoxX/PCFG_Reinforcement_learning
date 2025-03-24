import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set, Optional

# Constants
MAX_SENTENCE_LENGTH = 50
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_ATTENTION_HEADS = 8
NUM_TRANSFORMER_LAYERS = 6
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor for rewards
MCTS_SIMULATIONS = 100
EXPLORATION_CONSTANT = 1.0
TRAINING_EPISODES = 1000

class Rule:
    """
    Represents a grammar rule with a left-hand side (LHS) and right-hand side (RHS).
    For binary rules, RHS is a tuple of two symbols.
    For unary rules, RHS is a tuple with a single symbol.
    """
    def __init__(self, lhs: str, rhs: tuple, log_prob: float = 0.0):
        self.lhs = lhs
        self.rhs = rhs
        self.log_prob = log_prob  # Log probability of this rule (used in PCFG)
        self.is_binary = len(rhs) == 2
        self.is_unary = len(rhs) == 1
        self.is_terminal = self.is_unary and rhs[0][0].islower()  # Terminal symbols are lowercase
        self.rule_str = f"{lhs} -> {' '.join(rhs)}"
    
    def __str__(self):
        return self.rule_str
    
    def __repr__(self):
        return f"Rule({self.rule_str}, log_prob={self.log_prob:.4f})"
    
    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return self.lhs == other.lhs and self.rhs == other.rhs
    
    def __hash__(self):
        return hash((self.lhs, self.rhs))


class Grammar:
    """
    Represents a Probabilistic Context-Free Grammar (PCFG).
    """
    def __init__(self, non_terminals: Set[str], terminals: Set[str], start_symbol: str = "S"):
        self.non_terminals = non_terminals
        self.terminals = terminals
        self.start_symbol = start_symbol
        
        # Rules indexed by their left-hand side
        self.binary_rules = defaultdict(list)
        self.unary_rules = defaultdict(list)
        
        # All rules
        self.all_rules = set()
        
        # Pre-terminals (non-terminals that produce terminals)
        self.pre_terminals = set()
    
    def add_rule(self, rule: Rule):
        """Add a rule to the grammar."""
        self.all_rules.add(rule)
        if rule.is_binary:
            self.binary_rules[rule.lhs].append(rule)
        elif rule.is_unary:
            self.unary_rules[rule.lhs].append(rule)
            if rule.is_terminal:
                self.pre_terminals.add(rule.lhs)
    
    def get_binary_rules_for(self, lhs: str) -> List[Rule]:
        """Get all binary rules with the given LHS."""
        return self.binary_rules[lhs]
    
    def get_unary_rules_for(self, lhs: str) -> List[Rule]:
        """Get all unary rules with the given LHS."""
        return self.unary_rules[lhs]
    
    def get_rule_log_prob(self, rule: Rule) -> float:
        """Get the log probability of a rule."""
        if rule in self.all_rules:
            matching_rule = next(r for r in self.all_rules if r == rule)
            return matching_rule.log_prob
        return float('-inf')
    
    def normalize_rule_probabilities(self):
        """Normalize rule probabilities so they sum to 1 for each LHS."""
        for lhs in self.non_terminals:
            # Normalize binary rules
            binary_rules = self.get_binary_rules_for(lhs)
            if binary_rules:
                total = sum(math.exp(rule.log_prob) for rule in binary_rules)
                if total > 0:
                    for rule in binary_rules:
                        rule.log_prob = math.log(math.exp(rule.log_prob) / total)
            
            # Normalize unary rules
            unary_rules = self.get_unary_rules_for(lhs)
            if unary_rules:
                total = sum(math.exp(rule.log_prob) for rule in unary_rules)
                if total > 0:
                    for rule in unary_rules:
                        rule.log_prob = math.log(math.exp(rule.log_prob) / total)
    
    def sample_rule(self, lhs: str, is_terminal: bool = False) -> Optional[Rule]:
        """Sample a rule with the given LHS based on rule probabilities."""
        if is_terminal:
            rules = self.get_unary_rules_for(lhs)
            if not rules:
                return None
        else:
            rules = self.get_binary_rules_for(lhs) + self.get_unary_rules_for(lhs)
            if not rules:
                return None
        
        # Convert log probabilities to probabilities
        probs = [math.exp(rule.log_prob) for rule in rules]
        total = sum(probs)
        if total == 0:
            return random.choice(rules)
        
        probs = [p / total for p in probs]
        return np.random.choice(rules, p=probs)
    
    def generate_sentence(self, max_length: int = MAX_SENTENCE_LENGTH) -> List[str]:
        """Generate a random sentence from the grammar."""
        words = []
        
        def expand(symbol: str):
            if len(words) >= max_length:
                return
            
            if symbol in self.terminals:
                words.append(symbol)
                return
            
            if symbol in self.pre_terminals:
                rule = self.sample_rule(symbol, is_terminal=True)
                if rule:
                    words.append(rule.rhs[0])
                return
            
            rule = self.sample_rule(symbol)
            if not rule:
                return
            
            if rule.is_unary:
                expand(rule.rhs[0])
            else:  # binary rule
                expand(rule.rhs[0])
                expand(rule.rhs[1])
        
        expand(self.start_symbol)
        return words
    
    def __str__(self):
        result = f"Grammar with {len(self.non_terminals)} non-terminals and {len(self.terminals)} terminals\n"
        result += f"Start symbol: {self.start_symbol}\n"
        result += "Rules:\n"
        for rule in sorted(self.all_rules, key=lambda r: (r.lhs, r.rule_str)):
            prob = math.exp(rule.log_prob)
            result += f"  {rule.rule_str} [{prob:.4f}]\n"
        return result


class SentenceEncoder(nn.Module):
    """
    Transformer-based sentence encoder that converts a sentence into a fixed-size representation.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = EMBEDDING_DIM, 
                 hidden_dim: int = HIDDEN_DIM, num_heads: int = NUM_ATTENTION_HEADS, 
                 num_layers: int = NUM_TRANSFORMER_LAYERS, dropout: float = DROPOUT_RATE):
        super(SentenceEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(MAX_SENTENCE_LENGTH, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            attention_mask: Tensor of shape (batch_size, seq_len) with 1s for real tokens and 0s for padding
        Returns:
            sentence_encoding: Tensor of shape (batch_size, embedding_dim)
        """
        batch_size, seq_len = token_ids.shape
        
        # Create position IDs
        positions = torch.arange(seq_len, dtype=torch.long, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeddings = self.embedding(token_ids)
        position_embeddings = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Create attention mask for transformer (1 = ignore, 0 = attend)
        transformer_mask = (~attention_mask.bool()).float() * -1e9
        transformer_mask = transformer_mask.unsqueeze(1).unsqueeze(2)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(embeddings)
        
        # Use [CLS] token (first token) as sentence representation
        sentence_encoding = transformer_output[:, 0, :]
        
        # Process through output layer
        sentence_encoding = self.output_layer(sentence_encoding)
        
        return sentence_encoding


class PolicyNetwork(nn.Module):
    """
    Policy network that takes a sentence encoding and outputs parsing rule probabilities.
    """
    def __init__(self, embedding_dim: int, num_rules: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, num_rules)
    
    def forward(self, sentence_encoding: torch.Tensor):
        """
        Args:
            sentence_encoding: Tensor of shape (batch_size, embedding_dim)
        Returns:
            rule_logits: Tensor of shape (batch_size, num_rules)
        """
        x = F.relu(self.fc1(sentence_encoding))
        x = F.relu(self.fc2(x))
        rule_logits = self.fc3(x)
        return rule_logits


class Vocabulary:
    """
    Manages token-to-index and index-to-token mappings.
    """
    def __init__(self):
        self.token_to_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_to_token = {0: "<PAD>", 1: "<UNK>"}
        self.size = 2
    
    def add_token(self, token: str):
        if token not in self.token_to_index:
            self.token_to_index[token] = self.size
            self.index_to_token[self.size] = token
            self.size += 1
    
    def get_index(self, token: str) -> int:
        return self.token_to_index.get(token, self.token_to_index["<UNK>"])
    
    def get_token(self, index: int) -> str:
        return self.index_to_token.get(index, "<UNK>")
    
    def encode_sentence(self, sentence: List[str]) -> List[int]:
        return [self.get_index(token) for token in sentence]
    
    def decode_sentence(self, indices: List[int]) -> List[str]:
        return [self.get_token(index) for index in indices]


class ParseTree:
    """
    Represents a parse tree for a sentence.
    """
    def __init__(self, symbol: str, children=None):
        self.symbol = symbol  # Non-terminal or terminal symbol
        self.children = children if children is not None else []
        self.score = 0.0  # Used for evaluating the quality of the parse
        self.rule = None  # The rule that generated this node
    
    def is_terminal(self):
        return len(self.children) == 0
    
    def is_binary(self):
        return len(self.children) == 2
    
    def is_unary(self):
        return len(self.children) == 1
    
    def get_words(self) -> List[str]:
        """Get all terminal symbols (words) in the parse tree."""
        if self.is_terminal():
            return [self.symbol]
        
        words = []
        for child in self.children:
            words.extend(child.get_words())
        return words
    
    def __str__(self):
        if self.is_terminal():
            return self.symbol
        
        if self.is_unary():
            return f"({self.symbol} {str(self.children[0])})"
        
        return f"({self.symbol} {str(self.children[0])} {str(self.children[1])})"


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search for parsing.
    """
    def __init__(self, state: Tuple[int, int, str], parent=None):
        self.state = state  # (start_index, end_index, symbol)
        self.parent = parent
        self.children = {}  # Maps actions to child nodes
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []  # List of actions not yet tried from this state
    
    def add_child(self, action: Tuple, child_state: Tuple[int, int, str]) -> 'MCTSNode':
        """Add a child node with the given action and state."""
        child = MCTSNode(child_state, parent=self)
        self.children[action] = child
        return child
    
    def update(self, reward: float):
        """Update node statistics with the given reward."""
        self.visits += 1
        self.value += (reward - self.value) / self.visits
    
    def get_uct_value(self, parent_visits: int, exploration_constant: float) -> float:
        """Calculate the UCT value for this node."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = EXPLORATION_CONSTANT) -> Tuple[Tuple, 'MCTSNode']:
        """Select the best child node according to UCT formula."""
        if not self.children:
            return None, None
        
        best_action = None
        best_node = None
        best_value = float('-inf')
        
        for action, child in self.children.items():
            uct_value = child.get_uct_value(self.visits, exploration_constant)
            if uct_value > best_value:
                best_value = uct_value
                best_action = action
                best_node = child
        
        return best_action, best_node


class PCFGParser:
    """
    Probabilistic Context-Free Grammar parser using Monte Carlo Tree Search.
    """
    def __init__(self, grammar: Grammar, policy_network: PolicyNetwork, sentence_encoder: SentenceEncoder, 
                 vocabulary: Vocabulary, device: torch.device):
        self.grammar = grammar
        self.policy_network = policy_network
        self.sentence_encoder = sentence_encoder
        self.vocabulary = vocabulary
        self.device = device
        
        # Index rules for the policy network
        self.rule_index = {}
        self.rules = []
        for i, rule in enumerate(grammar.all_rules):
            self.rule_index[rule] = i
            self.rules.append(rule)
    
    def get_rule_indices(self, rules: List[Rule]) -> List[int]:
        """Get indices for the given rules."""
        return [self.rule_index[rule] for rule in rules if rule in self.rule_index]
    
    def encode_sentence(self, sentence: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a sentence for input to the neural networks."""
        token_ids = self.vocabulary.encode_sentence(sentence)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask = (token_ids != 0).float()
        return token_ids, attention_mask
    
    def get_policy_distribution(self, sentence: List[str]) -> torch.Tensor:
        """Get policy distribution for the given sentence."""
        token_ids, attention_mask = self.encode_sentence(sentence)
        
        # Get sentence encoding
        with torch.no_grad():
            sentence_encoding = self.sentence_encoder(token_ids, attention_mask)
            rule_logits = self.policy_network(sentence_encoding)
            rule_probs = F.softmax(rule_logits, dim=-1)
        
        return rule_probs.squeeze(0)
    
    def mcts_search(self, sentence: List[str], num_simulations: int = MCTS_SIMULATIONS) -> ParseTree:
        """Perform Monte Carlo Tree Search to find the best parse tree."""
        n = len(sentence)
        
        # Initialize root node
        root = MCTSNode((0, n, self.grammar.start_symbol))
        
        # Get policy distribution
        policy_distribution = self.get_policy_distribution(sentence)
        
        # Perform simulations
        for _ in range(num_simulations):
            # Selection
            node = root
            state_path = [node.state]
            action_path = []
            
            while node.untried_actions == [] and node.children:
                action, node = node.best_child()
                action_path.append(action)
                state_path.append(node.state)
            
            # Expansion
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                start, end, symbol = node.state
                if action[0] == "binary":
                    rule, split_point = action[1], action[2]
                    left_child_state = (start, split_point, rule.rhs[0])
                    right_child_state = (split_point, end, rule.rhs[1])
                    
                    left_child = node.add_child(("left", rule, split_point), left_child_state)
                    right_child = node.add_child(("right", rule, split_point), right_child_state)
                    
                    # Set untried actions for the new nodes
                    left_child.untried_actions = self._get_actions(left_child_state, sentence)
                    right_child.untried_actions = self._get_actions(right_child_state, sentence)
                    
                    # Continue with the right child (arbitrarily chosen)
                    node = right_child
                    state_path.append(node.state)
                    action_path.append(("right", rule, split_point))
                else:  # unary rule
                    rule = action[1]
                    child_state = (start, end, rule.rhs[0])
                    child = node.add_child(("unary", rule), child_state)
                    child.untried_actions = self._get_actions(child_state, sentence)
                    
                    node = child
                    state_path.append(node.state)
                    action_path.append(("unary", rule))
            
            # Simulation
            current_state = node.state
            while not self._is_terminal_state(current_state, sentence):
                actions = self._get_actions(current_state, sentence)
                if not actions:
                    break
                
                # Use policy network to guide selection
                valid_rules = [a[1] for a in actions if a[0] == "unary"]
                valid_rule_indices = self.get_rule_indices(valid_rules)
                
                if valid_rule_indices:
                    rule_probs = policy_distribution[valid_rule_indices]
                    rule_idx = torch.multinomial(rule_probs, 1).item()
                    action = actions[rule_idx]
                else:
                    action = random.choice(actions)
                
                # Apply action
                start, end, symbol = current_state
                if action[0] == "binary":
                    rule, split_point = action[1], action[2]
                    # Arbitrarily choose right child
                    current_state = (split_point, end, rule.rhs[1])
                else:  # unary rule
                    rule = action[1]
                    current_state = (start, end, rule.rhs[0])
            
            # Backpropagation
            # Calculate reward based on how much of the sentence was parsed
            reward = self._calculate_reward(state_path, sentence)
            
            for node in reversed([root] + [root.children[a] for a in action_path]):
                node.update(reward)
        
        # Extract the best parse tree
        return self._extract_parse_tree(root, sentence)
    
    def _get_actions(self, state: Tuple[int, int, str], sentence: List[str]) -> List[Tuple]:
        """Get all possible actions from the given state."""
        start, end, symbol = state
        actions = []
        
        # Terminal case
        if end - start == 1 and symbol in self.grammar.pre_terminals:
            word = sentence[start]
            for rule in self.grammar.get_unary_rules_for(symbol):
                if rule.rhs[0] == word:
                    actions.append(("unary", rule))
            return actions
        
        # Binary rules
        if end - start > 1 and symbol in self.grammar.non_terminals:
            for rule in self.grammar.get_binary_rules_for(symbol):
                for split_point in range(start + 1, end):
                    actions.append(("binary", rule, split_point))
        
        # Unary rules
        for rule in self.grammar.get_unary_rules_for(symbol):
            if not rule.is_terminal:
                actions.append(("unary", rule))
        
        return actions
    
    def _is_terminal_state(self, state: Tuple[int, int, str], sentence: List[str]) -> bool:
        """Check if the given state is a terminal state."""
        start, end, symbol = state
        return (end - start == 1 and symbol == sentence[start]) or symbol in self.grammar.terminals
    
    def _calculate_reward(self, state_path: List[Tuple[int, int, str]], sentence: List[str]) -> float:
        """Calculate reward based on how much of the sentence was parsed."""
        # Check if we have a complete parse
        if len(state_path) == 1:
            return 0.0
        
        # Check if the parse is valid
        for i in range(len(state_path) - 1):
            start1, end1, symbol1 = state_path[i]
            start2, end2, symbol2 = state_path[i + 1]
            
            if not (start1 <= start2 and end2 <= end1):
                return 0.0
        
        # Calculate coverage
        covered = set()
        for start, end, _ in state_path:
            for i in range(start, end):
                covered.add(i)
        
        coverage = len(covered) / len(sentence)
        return coverage
    
    def _extract_parse_tree(self, root: MCTSNode, sentence: List[str]) -> ParseTree:
        """Extract the best parse tree from the MCTS search."""
        def build_tree(node: MCTSNode) -> ParseTree:
            start, end, symbol = node.state
            
            # Terminal case
            if end - start == 1 and symbol == sentence[start]:
                return ParseTree(symbol)
            
            # Find best child
            best_action = None
            best_child = None
            best_value = float('-inf')
            
            for action, child in node.children.items():
                if child.visits > 0 and child.value > best_value:
                    best_value = child.value
                    best_action = action
                    best_child = child
            
            if best_action is None:
                return ParseTree(symbol)
            
            tree = ParseTree(symbol)
            tree.score = best_child.value
            
            if best_action[0] == "unary":
                rule = best_action[1]
                tree.rule = rule
                tree.children = [build_tree(best_child)]
            else:  # binary rule
                rule = best_action[1]
                split_point = best_action[2]
                tree.rule = rule
                
                left_child_action = ("left", rule, split_point)
                right_child_action = ("right", rule, split_point)
                
                left_child = node.children.get(left_child_action)
                right_child = node.children.get(right_child_action)
                
                if left_child and right_child:
                    tree.children = [build_tree(left_child), build_tree(right_child)]
            
            return tree
        
        return build_tree(root)


class GrammarLearner:
    """
    Main class for grammar induction through reinforcement learning.
    """
    def __init__(self, corpus: List[List[str]], device: torch.device = None):
        self.corpus = corpus
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize vocabulary
        self.vocabulary = Vocabulary()
        for sentence in corpus:
            for token in sentence:
                self.vocabulary.add_token(token)
        
        # Initialize grammar
        self.grammar = self._initialize_grammar()
        
        # Initialize neural networks
        self.sentence_encoder = SentenceEncoder(
            vocab_size=self.vocabulary.size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_ATTENTION_HEADS,
            num_layers=NUM_TRANSFORMER_LAYERS,
            dropout=DROPOUT_RATE
        ).to(self.device)
        
        self.policy_network = PolicyNetwork(
            embedding_dim=EMBEDDING_DIM,
            num_rules=len(self.grammar.all_rules)
        ).to(self.device)
        
        # Initialize parser
        self.parser = PCFGParser(
            grammar=self.grammar,
            policy_network=self.policy_network,
            sentence_encoder=self.sentence_encoder,
            vocabulary=self.vocabulary,
            device=self.device
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.sentence_encoder.parameters()) + list(self.policy_network.parameters()),
            lr=LEARNING_RATE
        )
        
        # Memory for experience replay
        self.memory = deque(maxlen=10000)
    
    def _initialize_grammar(self) -> Grammar:
        """Initialize the grammar with basic rules."""
        # Extract unique tokens from the corpus
        unique_tokens = set()
        for sentence in self.corpus:
            unique_tokens.update(sentence)
        
        # Create non-terminals and terminals
        non_terminals = {"S", "NP", "VP", "PP", "DT", "NN", "VB", "IN", "JJ", "RB"}
        terminals = unique_tokens
        
        # Create pre-terminals
        pre_terminals = {"DT", "NN", "VB", "IN", "JJ", "RB"}
        
        # Create grammar
        grammar = Grammar(non_terminals, terminals, start_symbol="S")
        
        # Add basic rules with uniform probabilities
        # Binary rules
        for rule_str in [
            "S -> NP VP",
            "NP -> DT NN",
            "NP -> JJ NN",
            "VP -> VB NP",
            "VP -> VB PP",
            "PP -> IN NP"
        ]:
            lhs, rhs = rule_str.split(" -> ")
            rhs_symbols = tuple(rhs.split())
            rule = Rule(lhs, rhs_symbols, log_prob=math.log(1.0))
            grammar.add_rule(rule)
        
        # Unary rules for pre-terminals to terminals
        for pre_terminal in pre_terminals:
            for terminal in terminals:
                rule = Rule(pre_terminal, (terminal,), log_prob=math.log(1.0 / len(terminals)))
                grammar.add_rule(rule)
        
        # Normalize rule probabilities
        grammar.normalize_rule_probabilities()
        
        return grammar
    
    def collect_experiences(self, num_episodes: int = 10):
        """Collect experiences by parsing sentences using MCTS."""
        for _ in range(num_episodes):
            # Randomly select a sentence from the corpus
            sentence = random.choice(self.corpus)
            
            # Parse the sentence using MCTS
            parse_tree = self.parser.mcts_search(sentence)
            
            # Store the experience
            token_ids, attention_mask = self.parser.encode_sentence(sentence)
            
            # Extract rules used in the parse tree
            rules_used = self._extract_rules_from_tree(parse_tree)
            rule_indices = self.parser.get_rule_indices(rules_used)
            
            if rule_indices:
                # Create a sparse target vector
                target = torch.zeros(len(self.grammar.all_rules), device=self.device)
                for idx in rule_indices:
                    target[idx] = 1.0 / len(rule_indices)
                
                # Store the experience
                self.memory.append((token_ids, attention_mask, target, parse_tree.score))
    
    def _extract_rules_from_tree(self, tree: ParseTree) -> List[Rule]:
        """Extract all rules used in the parse tree."""
        rules = []
        if tree.rule:
            rules.append(tree.rule)
        
        for child in tree.children:
            rules.extend(self._extract_rules_from_tree(child))
        
        return rules
    
    def train_step(self):
        """Perform a single training step."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample a batch of experiences
        batch = random.sample(self.memory, BATCH_SIZE)
        token_ids, attention_masks, targets, rewards = zip(*batch)
        
        # Prepare batch data
        token_ids = torch.cat(token_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        targets = torch.stack(targets)
        rewards = torch.tensor(rewards, device=self.device)
        
        # Forward pass
        sentence_encodings = self.sentence_encoder(token_ids, attention_masks)
        rule_logits = self.policy_network(sentence_encodings)
        rule_probs = F.softmax(rule_logits, dim=-1)
        
        # Calculate loss
        # Use cross-entropy loss weighted by rewards
        log_probs = F.log_softmax(rule_logits, dim=-1)
        loss = -torch.sum(targets * log_probs * rewards.unsqueeze(1)) / BATCH_SIZE
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes: int = TRAINING_EPISODES):
        """Train the model for the given number of episodes."""
        losses = []
        
        for episode in range(num_episodes):
            # Collect experiences
            self.collect_experiences()
            
            # Train on collected experiences
            loss = self.train_step()
            losses.append(loss)
            
            # Periodically print progress
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}, Loss: {loss:.4f}")
            
            # Periodically evaluate
            if episode % 100 == 0:
                self.evaluate()
        
        return losses
    
    def evaluate(self, num_sentences: int = 10):
        """Evaluate the model on a subset of the corpus."""
        total_score = 0.0
        
        for _ in range(num_sentences):
            # Randomly select a sentence from the corpus
            sentence = random.choice(self.corpus)
            
            # Parse the sentence using MCTS
            parse_tree = self.parser.mcts_search(sentence)
            
            # Calculate score
            score = parse_tree.score
            total_score += score
        
        avg_score = total_score / num_sentences
        print(f"Evaluation: Average parse score: {avg_score:.4f}")
        
        return avg_score
    
    def parse_sentence(self, sentence: List[str]) -> ParseTree:
        """Parse a sentence using the trained model."""
        return self.parser.mcts_search(sentence)
    
    def update_grammar(self):
        """Update grammar rule probabilities based on the policy network."""
        # Get all rules in the grammar
        all_rules = list(self.grammar.all_rules)
        
        # Create a dummy sentence to get policy distribution
        dummy_sentence = ["<UNK>"]
        rule_probs = self.parser.get_policy_distribution(dummy_sentence)
        
        # Update rule probabilities
        for i, rule in enumerate(all_rules):
            if i < len(rule_probs):
                rule.log_prob = torch.log(rule_probs[i] + 1e-10).item()
        
        # Normalize rule probabilities
        self.grammar.normalize_rule_probabilities()
    
    def add_rules(self, new_rules: List[Rule]):
        """Add new rules to the grammar."""
        for rule in new_rules:
            self.grammar.add_rule(rule)
        
        # Update the policy network to accommodate new rules
        old_policy_network = self.policy_network
        self.policy_network = PolicyNetwork(
            embedding_dim=EMBEDDING_DIM,
            num_rules=len(self.grammar.all_rules)
        ).to(self.device)
        
        # Copy over weights for existing rules
        with torch.no_grad():
            self.policy_network.fc1.weight.copy_(old_policy_network.fc1.weight)
            self.policy_network.fc1.bias.copy_(old_policy_network.fc1.bias)
            self.policy_network.fc2.weight.copy_(old_policy_network.fc2.weight)
            self.policy_network.fc2.bias.copy_(old_policy_network.fc2.bias)
            
            # Copy over weights for existing rules
            num_old_rules = old_policy_network.fc3.weight.size(0)
            self.policy_network.fc3.weight[:num_old_rules].copy_(old_policy_network.fc3.weight)
            self.policy_network.fc3.bias[:num_old_rules].copy_(old_policy_network.fc3.bias)
        
        # Update the parser
        self.parser = PCFGParser(
            grammar=self.grammar,
            policy_network=self.policy_network,
            sentence_encoder=self.sentence_encoder,
            vocabulary=self.vocabulary,
            device=self.device
        )
        
        # Update optimizer
        self.optimizer = optim.Adam(
            list(self.sentence_encoder.parameters()) + list(self.policy_network.parameters()),
            lr=LEARNING_RATE
        )
    
    def discover_new_rules(self):
        """Discover new rules by analyzing parse failures."""
        new_rules = []
        
        for sentence in self.corpus:
            parse_tree = self.parser.mcts_search(sentence)
            
            # If parse score is low, try to discover new rules
            if parse_tree.score < 0.5:
                # Extract words from the sentence
                words = sentence
                
                # Create potential new rules
                for i in range(len(words) - 1):
                    # Create a new binary rule
                    lhs = "X"  # A new non-terminal
                    rhs = (words[i], words[i+1])
                    rule = Rule(lhs, rhs, log_prob=math.log(0.1))
                    new_rules.append(rule)
        
        # Add new rules to the grammar
        if new_rules:
            self.add_rules(new_rules)
            print(f"Added {len(new_rules)} new rules to the grammar.")
    
    def save_model(self, path: str):
        """Save the model to the given path."""
        state = {
            'sentence_encoder': self.sentence_encoder.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'vocabulary': self.vocabulary,
            'grammar': self.grammar
        }
        torch.save(state, path)
    
    def load_model(self, path: str):
        """Load the model from the given path."""
        state = torch.load(path, map_location=self.device)
        
        self.sentence_encoder.load_state_dict(state['sentence_encoder'])
        self.policy_network.load_state_dict(state['policy_network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.vocabulary = state['vocabulary']
        self.grammar = state['grammar']
        
        # Update the parser
        self.parser = PCFGParser(
            grammar=self.grammar,
            policy_network=self.policy_network,
            sentence_encoder=self.sentence_encoder,
            vocabulary=self.vocabulary,
            device=self.device
        )


def main():
    # Example usage
    # Load corpus
    corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "in", "the", "park"],
        ["she", "saw", "a", "bird", "with", "binoculars"],
        ["the", "book", "is", "on", "the", "table"],
        ["I", "like", "to", "eat", "pizza", "with", "friends"]
    ]
    
    # Create grammar learner
    learner = GrammarLearner(corpus)
    
    # Train the model
    losses = learner.train(num_episodes=100)
    
    # Evaluate the model
    avg_score = learner.evaluate()
    print(f"Final evaluation score: {avg_score:.4f}")
    
    # Parse a new sentence
    new_sentence = ["the", "cat", "ran", "on", "the", "table"]
    parse_tree = learner.parse_sentence(new_sentence)
    print(f"Parse tree: {parse_tree}")
    print(f"Parse score: {parse_tree.score:.4f}")
    
    # Save the model
    learner.save_model("pcfg_model.pt")


if __name__ == "__main__":
    main()
                                                   