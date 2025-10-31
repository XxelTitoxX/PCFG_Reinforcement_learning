from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from sequence_embedding import TagEmbedder, WordTagEmbedder, BertWordEmbedder, IndexWordEmbedder, InductionEmbedder, TransformerLayer, ConvEncoderLayer

logger = getLogger(__name__)

def retrieve_rule_mask(rule_mask: torch.Tensor, symbol_pair: torch.Tensor) -> torch.Tensor:
    """
    Given a rule mask and a symbol pair, retrieves the corresponding rule mask for the symbol pair.

    Args:
        rule_mask (torch.Tensor): Tensor of shape (action_dim, state_dim, state_dim)
        symbol_pair (torch.Tensor): Tensor of shape (batch_size, 2) containing symbol pairs

    Returns:
        torch.Tensor: Tensor of shape (batch_size, action_dim) with the corresponding rule mask values.
    """
    batch_size = symbol_pair.shape[0]
    action_dim = rule_mask.shape[0]
    rows = torch.arange(action_dim, device=rule_mask.device).unsqueeze(0).repeat(batch_size, 1)
    cols = symbol_pair.unsqueeze(1).repeat(1, action_dim, 1)
    rule_weights = rule_mask[rows, cols[:, :, 0], cols[:, :, 1]]  # (batch_size, action_dim)
    return rule_weights

def get_adjacent_pairs(sentences: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Extracts and concatenates adjacent embedding pairs from each sentence in the batch.

    Args:
        sentences (torch.Tensor): Tensor of shape (batch_size, max_seq_len, embedding_dim)
        sequence_lengths (torch.Tensor): Tensor of shape (batch_size,) containing the actual lengths

    Returns:
        torch.Tensor: Tensor of shape (sequence_lengths.sum() - batch_size, 2 * embedding_dim)
                      containing the concatenated adjacent pairs.
    """
    batch_size, max_seq_len, embedding_dim = sentences.shape
    output_pairs = []

    for i in range(batch_size):
        seq_len = sequence_lengths[i].item()
        if seq_len < 2:
            continue  # No adjacent pairs possible
        sentence = sentences[i, :seq_len]  # (seq_len, embedding_dim)
        first = sentence[:-1]  # (seq_len - 1, embedding_dim)
        second = sentence[1:]  # (seq_len - 1, embedding_dim)
        pairs = torch.cat([first, second], dim=1)  # (seq_len - 1, 2 * embedding_dim)
        output_pairs.append(pairs)

    return torch.cat(output_pairs, dim=0)

def map_scores_to_sentences(embedding_pairs_scores: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Maps flat embedding pair scores back to their corresponding sentence positions.

    Args:
        embedding_pairs_scores (torch.Tensor): Tensor of shape ((sequence_lengths - 1).sum(),)
        sequence_lengths (torch.Tensor): Tensor of shape (batch_size,)

    Returns:
        torch.Tensor: Tensor of shape (batch_size, sequence_lengths.max() - 1)
                      with scores placed in their sentence positions, padded with 0.
    """
    batch_size = sequence_lengths.shape[0]
    max_len_minus1 = sequence_lengths.max().item() - 1
    output = torch.full((batch_size, max_len_minus1), float('-inf'), dtype=embedding_pairs_scores.dtype, device=embedding_pairs_scores.device)

    idx = 0
    for i in range(batch_size):
        length = sequence_lengths[i].item()
        num_pairs = max(length - 1, 0)
        if num_pairs > 0:
            output[i, :num_pairs] = embedding_pairs_scores[idx:idx + num_pairs]
            idx += num_pairs

    return output
    

class ActorNetwork(nn.Module):
    def __init__(self, embedding_dim:int, action_dim:int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class PositionActor(nn.Module):
    def __init__(self, embedding_dim : int, temperature : float = 1.0):
        super().__init__()
        self.actor_network: ActorNetwork = ActorNetwork(embedding_dim=embedding_dim)
        self.temperature: float = temperature

    def forward(self, sequence_embedding: torch.Tensor, sentence_lengths: torch.Tensor):
        """
        Given a batch of sequence embeddings and ground truth positions, returns the log probabilities of the ground truth positions.
        :param sequence_embedding: torch.Tensor of shape (batch_size, seq_max_len, embedding_dim)
        :param sentence_lengths: torch.Tensor of shape (batch_size,)
        :param gt_positions: torch.Tensor of shape (batch_size,)
        :return: position_action_logprob: torch.Tensor of shape (batch_size,)
        """

        # Get adjacent pairs of embeddings
        adjacent_pairs: torch.Tensor = get_adjacent_pairs(sequence_embedding, sentence_lengths) # (num_pairs, 2 * embedding_dim)

        embedding_pairs_scores: torch.Tensor = self.actor_network(adjacent_pairs) # (num_pairs, 1)

        # Map scores back to sentence positions
        scores: torch.Tensor = map_scores_to_sentences(embedding_pairs_scores.squeeze(dim=1), sentence_lengths) # (batch_size, max_seq_len - 1)

        # Get position action distribution
        action_probs = torch.softmax(scores/self.temperature, dim=1) # (batch_size, max_seq_len - 1)

        # Get log probability of ground truth positions
        dist: Categorical = Categorical(action_probs)
        if self.training:
            position_action = dist.sample().long() # (batch_size,)
        else:
            position_action = torch.argmax(action_probs, dim=1) # (batch_size,)
        position_action_logprob: torch.Tensor = dist.log_prob(position_action) # (batch_size,)
        entropy = dist.entropy() # (batch_size,)

        return position_action, position_action_logprob, entropy
    
    def evaluate(self, sequence_embedding: torch.Tensor, sentence_lengths: torch.Tensor, position_action: torch.Tensor):
        # Get adjacent pairs of embeddings
        adjacent_pairs: torch.Tensor = get_adjacent_pairs(sequence_embedding, sentence_lengths) # (num_pairs, 2 * embedding_dim)

        embedding_pairs_scores: torch.Tensor = self.actor_network(adjacent_pairs) # (num_pairs, 1)

        # Map scores back to sentence positions
        scores: torch.Tensor = map_scores_to_sentences(embedding_pairs_scores.squeeze(dim=1), sentence_lengths) # (batch_size, max_seq_len - 1)

        # Get position action distribution
        action_probs = torch.softmax(scores/self.temperature, dim=1) # (batch_size, max_seq_len - 1)

        # Get log probability of ground truth positions
        dist: Categorical = Categorical(action_probs)
        position_action_logprob: torch.Tensor = dist.log_prob(position_action) # (batch_size,)
        entropy = dist.entropy() # (batch_size,)

        return position_action_logprob, entropy

    

class SymbolActor(nn.Module):
    def __init__(self, embedding_dim:int, action_dim:int, temperature:float = 1.0):
        super().__init__()
        self.actor_network: ActorNetwork = ActorNetwork(embedding_dim=embedding_dim, action_dim=action_dim)
        self.temperature = temperature

    def forward(self, sequence_embedding:torch.Tensor, position_action:torch.Tensor):
        batch_size = sequence_embedding.shape[0]

        # Concatenate embedding for the pair of constituents (position_action, position_action +1)
        symbol_pair_emb: torch.Tensor = torch.cat((sequence_embedding[torch.arange(batch_size), position_action], sequence_embedding[torch.arange(batch_size), position_action + 1]), dim=1) # (batch_size, embedding_dim*2)

        # Calculate symbol scores for each pair
        symbol_scores: torch.Tensor = self.actor_network(symbol_pair_emb) # (batch_size, action_dim)

        # Transform symbol logits into probabilities
        symbol_probs: torch.Tensor = torch.softmax(symbol_scores/self.temperature, dim=1) # (batch_size, action_dim)

        # Sample symbol action
        dist: Categorical = Categorical(symbol_probs)
        if self.training:
            symbol_action: torch.Tensor = dist.sample() # (batch_size,)
        else:
            symbol_action: torch.Tensor = torch.argmax(symbol_probs, dim=1) # (batch_size,)
        
        symbol_action_logprob: torch.Tensor = dist.log_prob(symbol_action) # (batch_size,)

        action_entropy = dist.entropy()

        return symbol_action, symbol_action_logprob, action_entropy
    
    def evaluate(self, sequence_embedding:torch.Tensor, position_action:torch.Tensor, symbol_action:torch.Tensor):
        batch_size = sequence_embedding.shape[0]

        # Concatenate embedding for the pair of constituents (position_action, position_action +1)
        symbol_pair_emb: torch.Tensor = torch.cat((sequence_embedding[torch.arange(batch_size), position_action], sequence_embedding[torch.arange(batch_size), position_action + 1]), dim=1) # (batch_size, embedding_dim*2)

        # Calculate symbol scores for each pair
        symbol_scores: torch.Tensor = self.actor_network(symbol_pair_emb) # (batch_size, action_dim)

        # Transform symbol logits into probabilities
        symbol_probs: torch.Tensor = torch.softmax(symbol_scores/self.temperature, dim=1) # (batch_size, action_dim)

        # Get action distributions
        dist: Categorical = Categorical(symbol_probs)
        
        symbol_action_logprob: torch.Tensor = dist.log_prob(symbol_action) # (batch_size,)
        entropy = dist.entropy()

        return symbol_action_logprob, entropy


    

class ActorCritic(nn.Module):
    def __init__(
            self, state_dim: int, embedding_dim : int, action_dim: int, n_layer: int, num_heads: int, vocab_size: int
    ):
        super().__init__()

        self.state_dim: int = state_dim
        self.embedding_dim: int = embedding_dim
        self.action_dim: int = action_dim
        self.n_layer: int = n_layer
        assert self.n_layer > 0, f"n_layer must be greater than 0, got {self.n_layer}"
        assert self.embedding_dim > 0, f"embedding_dim must be greater than 0, got {self.embedding_dim}"

        self.tag_embedder: TagEmbedder = TagEmbedder(tag_dim=state_dim, embedding_dim=embedding_dim)
        #self.word_embedder: IndexWordEmbedder = IndexWordEmbedder(vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.word_embedder: BertWordEmbedder = BertWordEmbedder(embedding_dim=embedding_dim)
        self.word_tag_embedder: WordTagEmbedder = WordTagEmbedder(
            word_embedder=self.word_embedder, tag_embedder=self.tag_embedder, embedding_dim=embedding_dim, num_nt=action_dim
        )
        self.induction_embedder: InductionEmbedder = InductionEmbedder(tag_embedding_dim=embedding_dim,
            embedding_dim=embedding_dim, n_layer=n_layer, no_tag=False
        )
        self.transformer_layer: ConvEncoderLayer = ConvEncoderLayer(embedding_dim=embedding_dim, num_layers=n_layer)
        #self.transformer_layer: TransformerLayer = TransformerLayer(embedding_dim=embedding_dim, num_layers=n_layer, num_heads=num_heads, dropout=0.1)

        self.position_actor: PositionActor = PositionActor(embedding_dim=embedding_dim)

        self.symbol_actor: SymbolActor = SymbolActor(embedding_dim=embedding_dim, action_dim=action_dim)

        # critic
        self.critic: nn.Sequential = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        logger.info(
            f"ActorCritic initialized with {sum(p.numel() for p in self.parameters()):,} parameters, "
            f"state_dim={state_dim}, action_dim={action_dim}, "
            f"embedding_dim={embedding_dim}, n_layer={n_layer}"
        )

    def encode_state(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Encode the state to a tensor of dimension H.
        :param state: torch.Tensor of shape (B, SEQ, E)
                where B is the batch size, SEQ, is the sequence max length, E is the embedding dim
        :return: cls_token, seq_embedding, both torch.Tensor of respective shape (B, E) and (B, SEQ, E), E standing for embedding dimension
        """
        assert len(state.shape) == 3, f"state must be a 3d tensor (batch_size, seq_max_len, embedding_dim), got {state.shape}"
        cls_token, sequence_embedding = self.transformer_layer(state, mask=mask)
        return cls_token, sequence_embedding
    
    def encode_sentence(self, batch_sentences) -> torch.Tensor:
        """
        Gives word-level embedding for a batch of sentences.
        batch_sentences: list[Sentence]
        Returns: torch.Tensor of shape (batch_size, seq_max_len, embedding_dim)
        """
        return self.word_tag_embedder(batch_sentences)
    
    def forward(self):
        raise NotImplementedError("ActorCritic does not implement forward method. Use act or evaluate instead.")
        

    def act(self, states: torch.Tensor, mask: torch.Tensor, position_update:torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a batch of states, sample an action for each state, calculate the action log probability, and new state value.
        :param state: torch.Tensor of shape (batch_size, seq_max_len, embedding_dim)
        :param mask: torch.Tensor of shape (batch_size, seq_max_len) where 1 means the position is valid and 0 means it is padding
        :return: tuple of action, action log probability, and state value, all are np.ndarray of shape (batch_size,)
        """
        batch_size: int = states.shape[0]
        sentence_lengths: torch.Tensor = torch.sum(mask, dim=1) # (batch_size,)
        assert len(states.shape) == 3, (
            f"state must be a 3d tensor (batch_size, seq_max_len, embedding_dim), got {states.shape}"
        )

        cls_token, sequence_embedding = self.encode_state(
            states, mask
        ) # cls_token: (B, E), sequence_embedding: (B, SEQ, E)
        assert cls_token.shape == (batch_size, self.embedding_dim), (
            f"cls_token must have a shape ({batch_size},{self.embedding_dim}), got {cls_token.shape}"
        )
        assert sequence_embedding.shape == (batch_size, states.shape[1], self.embedding_dim), ( 
            f"sequence_embedding must have a shape ({batch_size},{states.shape[1]},{self.embedding_dim}), got {sequence_embedding.shape}"
        )

        position_action, position_action_logprob, _ = self.position_actor(sequence_embedding, sentence_lengths)
        if position_update is not None: # supervised position update
            pos_update = position_update
        else:
            pos_update = position_action
        symbol_action, symbol_action_logprob, _ = self.symbol_actor(sequence_embedding, pos_update)

        # state_val: (B,1)
        state_val: torch.Tensor = self.critic(cls_token).squeeze(dim=1)
        assert position_action.shape == position_action_logprob.shape == symbol_action.shape == symbol_action_logprob.shape == state_val.shape == (batch_size,), (
            f"position_action, position_action_logprob, symbol_action, symbol_action_logprob, and state_val must have a shape (batch_size,), "
            f"got {position_action.shape}, {position_action_logprob.shape}, {symbol_action.shape}, {symbol_action_logprob.shape}, {state_val.shape}"
        )
        assert torch.all(position_action >= 0) and torch.all(position_action < sentence_lengths-1), (
            f"action must be in range [0, sentence_length-2), got {position_action.min()}, {position_action.max()}"
        )
        assert torch.all(position_action_logprob > float('-inf')), (
            f"action_logprob must be greater than -inf"
        )
        
        return position_action.to(torch.int32), position_action_logprob, symbol_action.to(torch.int32), symbol_action_logprob, state_val

    def evaluate(
            self, states: torch.Tensor, mask: torch.Tensor, position_action: torch.Tensor, position_update: torch.Tensor, symbol_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a batch of state and action, calculate the action log probability, entropy, and state value.
        :param state: torch.Tensor of shape (batch_size, seq_max_len)
        :param position_action: torch.Tensor of shape (batch_size,) the sampled action from position actor
        :param position_update: torch.Tensor of shape (batch_size,) the position used for the constituent merge (can be ground truth or sampled)
        :param symbol_action: torch.Tensor of shape (batch_size,)
        :return: tuple of action log probability, entropy, and state value, all are torch.Tensor of shape (batch_size,)
        """
        batch_size: int = states.shape[0]
        sentence_lengths: torch.Tensor = torch.sum(mask, dim=1) # (batch_size,)
        assert len(states.shape) == 3, (
            f"state must be a 3d tensor (batch_size, seq_max_len, embedding_dim), got {states.shape}"
        )

        cls_token, sequence_embedding = self.encode_state(
            states, mask
        ) # cls_token: (B, E), sequence_embedding: (B, SEQ, E)
        assert cls_token.shape == (batch_size, self.embedding_dim), (
            f"cls_token must have a shape ({batch_size},{self.embedding_dim}), got {cls_token.shape}"
        )
        assert sequence_embedding.shape == (batch_size, states.shape[1], self.embedding_dim), ( 
            f"sequence_embedding must have a shape ({batch_size},{states.shape[1]},{self.embedding_dim}), got {sequence_embedding.shape}"
        )

        position_action_logprob, position_entropy = self.position_actor.evaluate(sequence_embedding, sentence_lengths, position_action)
        symbol_action_logprob, symbol_entropy = self.symbol_actor.evaluate(sequence_embedding, position_update, symbol_action)

        # state_val: (B,1)
        state_val: torch.Tensor = self.critic(cls_token).squeeze(dim=1)
        assert position_action.shape == position_action_logprob.shape == symbol_action.shape == symbol_action_logprob.shape == state_val.shape == (batch_size,), (
            f"position_action, position_action_logprob, symbol_action, symbol_action_logprob, and state_val must have a shape (batch_size,), "
            f"got {position_action.shape}, {position_action_logprob.shape}, {symbol_action.shape}, {symbol_action_logprob.shape}, {state_val.shape}"
        )
        assert torch.all(position_action >= 0) and torch.all(position_action < sentence_lengths-1), (
            f"action must be in range [0, sentence_length-2), got {position_action.min()}, {position_action.max()}"
        )
        assert torch.all(position_action_logprob > float('-inf')), (
            f"action_logprob must be greater than -inf"
        )

        return position_action_logprob, position_entropy, symbol_action_logprob, symbol_entropy, state_val
    
    def state_val(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of state, calculate the state value from the critic network.
        :param state: torch.Tensor of shape (batch_size, seq_max_len, enbedding_dim)
        :return: state value, torch.Tensor of shape (batch_size,)
        """
        batch_size: int = states.shape[0]
        assert len(states.shape) == 2, (
            f"state must be a 2d tensor (batch_size, seq_max_len), got {states.shape}"
        )

        cls_token, _ = self.encode_state(
            states, mask
        ) # cls_token: (B, E), sequence_embedding: (B, SEQ, E)
        assert cls_token.shape == (batch_size, self.embedding_dim), (
            f"cls_token must have a shape ({batch_size},{self.embedding_dim}), got {cls_token.shape}"
        )
        # state_val: (B,1)
        state_val: torch.Tensor = self.critic(cls_token).squeeze(dim=1)
        return state_val
    
    def fuse_constituents(self, tag: torch.Tensor, left: torch.Tensor, right: torch.Tensor, dropout:bool=False) -> torch.Tensor:
        """
        Fuses the tag with the left and right constituents to create a new embedding.
        :param tag: torch.Tensor of shape (batch_size, embedding_dim)
        :param left: torch.Tensor of shape (batch_size, embedding_dim)
        :param right: torch.Tensor of shape (batch_size, embedding_dim)
        :return: torch.Tensor of shape (batch_size, embedding_dim)
        """
        assert left.shape == right.shape == (tag.shape[0], self.embedding_dim), (
            f"left, and right must have a shape ({tag.shape[0]}, {self.embedding_dim}), got {left.shape}, {right.shape}"
        )
        #return tag + 0.4 * (left + right)  # Linear combination
        #return tag
        return self.induction_embedder(tag, left, right, dropout=dropout)
    
    def get_cls_token(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the cls token for the given states.
        :param states: torch.Tensor of shape (batch_size, seq_max_len, embedding_dim)
        :param mask: torch.Tensor of shape (batch_size, seq_max_len) where 1 means the position is valid and 0 means it is padding
        :return: cls_token, torch.Tensor of shape (batch_size, embedding_dim)
        """
        assert (states.shape[0] == mask.shape[0]) and (states.shape[1] == mask.shape[1]), (
            f"states and mask must have the same shape, got {states.shape} and {mask.shape}")
        cls_token, _ = self.encode_state(states, mask)
        return cls_token
        
    
    def copy(self) -> "ActorCritic":
        """
        Creates a copy of the current ActorCritic instance with the same weights.

        :return: A new ActorCritic instance with identical weights.
        """
        # Create a new model with the same architecture
        copied_model = ActorCritic(
            state_dim=self.state_dim, 
            embedding_dim=self.embedding_dim,
            action_dim=self.action_dim, 
            n_layer=self.n_layer  # Assuming n_layer is the number of layers in state_encoder
        )

        # Copy weights from the original model
        copied_model.load_state_dict(self.state_dict())

        # Set the copied model to evaluation mode
        copied_model.eval()

        return copied_model
