from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from sequence_embedding import LSTMEmbedding, EnhancedTransformerEmbedding

logger = getLogger(__name__)


class ActorCritic(nn.Module):
    def __init__(
            self, state_dim: int, embedding_dim : int, seq_n_layers : int, hidden_dim : int, action_dim: int, n_layer: int
    ):
        super().__init__()

        self.state_dim: int = state_dim
        self.embedding_dim: int = embedding_dim
        self.seq_n_layers: int = seq_n_layers
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.n_layer: int = n_layer
        assert hidden_dim > 1, f"hidden_dim must be greater than 1, got {hidden_dim}"

        self.state_encoder: EnhancedTransformerEmbedding = EnhancedTransformerEmbedding(state_dim=state_dim, embedding_dim=embedding_dim,
                                                          hidden_dim=hidden_dim, num_layers=seq_n_layers,
                                                          #bidirectional=True, dropout=0.1
                                                          )

        # discrete actor
        self.actor: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        # critic
        self.critic: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

        logger.info(
            f"ActorCritic initialized with {sum(p.numel() for p in self.parameters()):,} parameters, "
            f"state_dim={state_dim}, action_dim={action_dim}, "
            f"hidden_dim={hidden_dim}, n_layer={n_layer}"
        )

    def forward(self):
        raise NotImplementedError

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode the state to a tensor of dimension H.
        :param state: torch.Tensor of shape (B, SEQ)
                where B is the batch size, SEQ, is the sequence max length
        :return: torch.Tensor of shape (B, H) where H is the hidden dimension
        """
        assert len(state.shape) == 2, f"state must be a 3d tensor (batch_size, seq_max_len), got {state.shape}"
        assert torch.all((state >= -1) & (state < self.state_dim)), f"state must be in range [0, state_dim), got {state.min()}, {state.max()}"
        state_encoded: torch.Tensor = self.state_encoder(state)
        return state_encoded

    def act(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a batch of states, sample an action for each state, calculate the action log probability, and new state value.
        :param state: np.ndarray of shape (batch_size, seq_max_len) where each value is a symbol index (-1 is padding value, shifted by 1 when given to state_encoder)
        :return: tuple of action, action log probability, and state value, all are np.ndarray of shape (1)
        """

        batch_size: int = states.shape[0]
        assert len(states.shape) == 2, (
            f"state must be a 3d tensor (batch_size, seq_max_len), got {states.shape}"
        )

        state_encoded: torch.Tensor = self.encode_state(
            states
        )
        assert state_encoded.shape == (batch_size, self.hidden_dim), (
            f"state_encoded must have a shape ({batch_size},{self.hidden_dim}), got {state_encoded.shape}"
        )

        # action_probs: (B,A) where A is the number of actions
        action_probs: torch.Tensor = self.actor(state_encoded)
        # state_val: (B,1)
        state_val: torch.Tensor = self.critic(state_encoded).squeeze(dim=1)

        dist: Categorical = Categorical(action_probs)
        action: torch.Tensor = dist.sample()
        action_logprob: torch.Tensor = dist.log_prob(action)

        assert action.shape == action_logprob.shape == state_val.shape == (batch_size,), (
            f"action, action_logprob, and log_prob must have a shape (batch_size,), "
            f"got {action.shape}, {action_logprob.shape}, {state_val.shape}"
        )
        return action, action_logprob, state_val

    def evaluate(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a batch of state and action, calculate the action log probability, entropy, and state value.
        :param state: torch.Tensor of shape (batch_size, seq_max_len)
        :param action: torch.Tensor of shape (batch_size,)
        :return: tuple of action log probability, entropy, and state value, all are torch.Tensor of shape (batch_size,)
        """
        assert action.shape == (state.shape[0],), (
            f"actions must have a shape ({state.shape[0]},), got {action.shape}"
        )
        assert torch.all(action >= 0) and torch.all(action < self.action_dim), f"actions must be in range [0, {self.action_dim}), got {action.min()}, {action.max()}"

        # state_encoded: (B, H)
        state_encoded: torch.Tensor = self.encode_state(state)

        # action_probs: (B, A) where A is the number of actions
        action_probs: torch.Tensor = self.actor(state_encoded)
        # state_val: (B, 1)
        state_val: torch.Tensor = self.critic(state_encoded)

        dist: Categorical = Categorical(action_probs)
        action_logprobs: torch.Tensor = dist.log_prob(action)
        dist_entropy: torch.Tensor = dist.entropy()

        return action_logprobs, dist_entropy, state_val.squeeze(dim=1)
    
    def copy(self) -> "ActorCritic":
        """
        Creates a copy of the current ActorCritic instance with the same weights.

        :return: A new ActorCritic instance with identical weights.
        """
        # Create a new model with the same architecture
        copied_model = ActorCritic(
            state_dim=self.state_dim, 
            embedding_dim=self.embedding_dim,
            seq_n_layers=self.seq_n_layers,
            action_dim=self.action_dim, 
            hidden_dim=self.hidden_dim, 
            n_layer=self.n_layer  # Assuming n_layer is the number of layers in state_encoder
        )

        # Copy weights from the original model
        copied_model.load_state_dict(self.state_dict())

        # Set the copied model to evaluation mode
        copied_model.eval()

        return copied_model
