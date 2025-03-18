from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

logger = getLogger(__name__)


class FullyConnected(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.mlp(x)
        x = self.ln(x)
        return x


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, n_layer: int):
        super().__init__()
        assert n_layer > 1, f"n_layer must be greater than 1, got {n_layer}"
        self.state_dim: int = state_dim
        self.hidden_dim: int = hidden_dim

        self.input = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.blocks = nn.Sequential(
            *[FullyConnected(hidden_dim)
              for _ in range(n_layer - 1)]
        )

    def forward(
            self, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a state to a tensor of dimension H.
        :param state: torch.Tensor of shape (B, R)
                    where B is batch size, R is the number of total rules (=state_dim)
        :return: torch.Tensor of shape (B, H) where H is the hidden dimension
        """
        assert len(state.shape) == 2, f"state must be a 2d tensor, got {state.shape}"
        B, _ = state.shape
        assert state.shape == (B, self.state_dim), (
            f"state must have a shape ({B}, {self.state_dim}), got {state.shape}"
        )
        assert state.isnan().sum().item() == 0, (
            f"state must not have NaN, got {state} with {state.isnan().sum().item()} NaN"
        )
        assert state.isinf().sum().item() == 0, (
            f"state must not have inf, got {state} with {state.isinf().sum().item()} inf"
        )

        X: torch.Tensor = self.input(state)
        X = self.blocks(X)

        assert X.shape == (B, self.hidden_dim), (
            f"X must have a shape ({B}, {self.hidden_dim}), got {X.shape}"
        )
        assert X.isnan().sum().item() == 0, (
            f"X must not have NaN, got {X} with {X.isnan().sum().item()} NaN"
        )
        return X


class ActorCritic(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int, n_layer: int
    ):
        super().__init__()

        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        assert hidden_dim > 1, f"hidden_dim must be greater than 1, got {hidden_dim}"

        self.state_encoder: StateEncoder = StateEncoder(
            state_dim, hidden_dim, n_layer
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
        :param state: torch.Tensor of shape (B, R)
                where B is the batch size, and R is the number of total rules (=state_dim)
        :return: torch.Tensor of shape (B, H) where H is the hidden dimension
        """
        assert len(state.shape) == 2, f"state must be a 2d tensor, got {state.shape}"
        state_encoded: torch.Tensor = self.state_encoder(state)
        return state_encoded

    def act(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given a state, sample an action, calculate the action log probability, and state value.
        :param state: np.ndarray of shape (R) where R is the number of total rules (=state_dim)
        :return: tuple of action, action log probability, and state value, all are np.ndarray of shape (1)
        """
        assert len(state.shape) == 1, (
            f"state_encoded must be a 1d np.ndarray, got {state.shape}"
        )

        state_encoded: torch.Tensor = self.encode_state(
            torch.from_numpy(state).unsqueeze(dim=0)
        ).squeeze(dim=0)
        assert state_encoded.shape == (self.hidden_dim,), (
            f"state_encoded must have a shape ({self.hidden_dim},), got {state_encoded.shape}"
        )

        # action_probs: (A,) where A is the number of actions
        action_probs: torch.Tensor = self.actor(state_encoded)
        # state_val: (1,)
        state_val: torch.Tensor = self.critic(state_encoded)

        dist: Categorical = Categorical(action_probs)
        action: torch.Tensor = dist.sample().reshape(1)
        action_logprob: torch.Tensor = dist.log_prob(action)

        assert action.shape == action_logprob.shape == state_val.shape == (1,), (
            f"action, action_logprob, and log_prob must have a shape (1,), "
            f"got {action.shape}, {action_logprob.shape}, {state_val.shape}"
        )
        return action.numpy(force=True), action_logprob.numpy(force=True), state_val.numpy(force=True)

    def evaluate(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a batch of state and action, calculate the action log probability, entropy, and state value.
        :param state: torch.Tensor of shape (B, R)
                    where B is the batch size, and R is the number of total rules (=state_dim)
        :param action: torch.Tensor of shape (B)
        :return: tuple of action log probability, entropy, and state value, all are torch.Tensor of shape (B)
        """
        assert action.shape == (state.shape[0],), (
            f"actions must have a shape ({state.shape[0]},), got {action.shape}"
        )

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
