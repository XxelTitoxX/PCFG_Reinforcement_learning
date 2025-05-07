import time
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Optional

import jsbeautifier
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses_json import dataclass_json
from torch.optim import Adam

from actor_critic import ActorCritic
from grammar_env.corpus.corpus import Corpus
from env import Environment
from result_saver import ResultSaver
from writer import Writer

logger = getLogger(__name__)

# https://discuss.pytorch.org/t/cpu-usage-far-too-high-and-training-inefficient/57228/3
# https://discuss.pytorch.org/t/pytorch-v2-high-cpu-consumption/205990
# OpenMP uses all the cpu cores by default, leading to inefficiency.
torch.set_num_threads(1)

PURE_REINFORCE = True

def mean(x: list[Any]) -> float:
    return sum(x) / len(x)


@dataclass(frozen=True)
class RolloutBuffer:
    """
    Batch of data collected using simulation.
    Used in the PPO algorithm.
    """
    obs: np.ndarray
    """
    np.ndarray of shape (number of timesteps, R) where R is the number of total rules.
    
    The observations collected this batch. Basically a batch of states.
    """
    acts: np.ndarray
    """
    np.ndarray of shape (number of timesteps)
    
    The actions collected this batch.
    """
    log_probs: np.ndarray
    """
    np.ndarray of shape (number of timesteps)
    
    The log probabilities of each action taken this batch.
    """
    rtgs: np.ndarray
    """
    np.ndarray of shape (number of timesteps)
    
    The Rewards-To-Go of each timestep in this batch.
    """
    lens: list[int]
    """
    The lengths of each episode this batch.
    """

    def __post_init__(self):
        assert (
                (sum(self.lens),) == (self.obs.shape[0],) == self.acts.shape
                == self.log_probs.shape == self.rtgs.shape
        ), (
            f"Shapes of Tensors are wrong, "
            f"got {(sum(self.lens),)} == {(self.obs.shape[0],)} == {self.acts.shape} == "
            f"{self.log_probs.shape} == {self.rtgs.shape}"
        )
        assert np.isnan(self.obs).sum().item() == 0, f"obs must not have NaN, got {self.obs}"
        assert np.isinf(self.obs).sum().item() == 0, f"obs must not have inf, got {self.obs}"
        assert np.isnan(self.acts).sum().item() == 0, f"acts must not have NaN, got {self.acts}"
        assert np.isinf(self.acts).sum().item() == 0, f"acts must not have inf, got {self.acts}"
        assert np.isnan(self.log_probs).sum().item() == 0, f"log_probs must not have NaN, got {self.log_probs}"
        assert np.isinf(self.log_probs).sum().item() == 0, f"log_probs must not have inf, got {self.log_probs}"
        assert np.isnan(self.rtgs).sum().item() == 0, f"rtgs must not have NaN, got {self.rtgs}"
        assert np.isinf(self.rtgs).sum().item() == 0, f"rtgs must not have inf, got {self.rtgs}"


@dataclass_json
@dataclass
class PPOConfig:
    # Grammar parameters
    num_non_terminals: int = 6  # Number of non-terminals

    # Criterion parameters
    criterion: str = "f1"  # Criterion to use for training
    num_sentences_per_score: int = 256  # Number of sentences used to score per criterion
    num_sentences_per_batch: int = 32  # Number of sentences to process per batch
    num_epochs: int = 50

    # Algorithm parameters
    n_updates_per_iteration: int = 10  # Number of times to update actor/critic per iteration
    lr: float = 2e-4  # Learning rate of optimizer
    gamma: float = 0.9  # Discount factor to be applied when calculating Rewards-To-Go
    clip: float = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
    actor_weight: float = 1.  # Weight of the actor loss
    critic_weight: float = 0.5  # Weight of the critic loss
    entropy_weight: float = 0.05  # Weight of the entropy loss
    entropy_weight_decay: float = 0.98  # Decay of the entropy weight
    entropy_weight_min: float = 0.01  # Minimum entropy weight
    entropy_weight_decay_freq: int = 10  # How often to decay the entropy weight

    max_num_steps: int = 100  # Maximum number of steps to run in the environment

    # Miscellaneous parameters
    save_freq: int = 20  # How often we save in number of iterations
    seed: int = 0  # Sets the seed of our program, used for reproducibility of results
    min_ep_rews_threshold: float = 0.  # Minimum episodic rewards threshold to log the grammar

    # Network parameters
    n_layer: int = 1
    n_head: int = 2
    embedding_dim: int = 64

    gradient_clip : Optional[float] = None


class PPO:
    def __init__(
            self, train_corpus: Corpus, valid_corpus: Corpus, persistent_dir: Path,
            writer: Writer, device: torch.device,
            config: PPOConfig
    ):
        self.train_corpus: Corpus = train_corpus
        self.valid_corpus: Corpus = valid_corpus
        self.persistent_dir: Path = persistent_dir
        self.writer: Writer = writer
        self.device: torch.device = device
        self.config = config
        self.state_dim: int = config.num_non_terminals + 14

        torch.manual_seed(config.seed)

        self.train_dataloader = train_corpus.get_dataloader(config.num_sentences_per_batch)
        self.valid_dataloader = valid_corpus.get_dataloader(config.num_sentences_per_score)

        self.result_saver: ResultSaver = ResultSaver(
            self.persistent_dir, self.writer,
            self.train_corpus, self.valid_corpus, self.device,
            config.num_sentences_per_score, config.max_num_steps,
        )

        # Setup environment
        """
        match config.criterion:
            case "prob":
                self._criterion: Criterion = ProbabilityCriterion(
                    self.train_corpus, self.device
                )
            case "cov":
                self._criterion: Criterion = CoverageCriterion(
                    self.train_corpus, self.device
                )
            case "f1":
                self._criterion: Criterion = F1Criterion(
                    self.train_corpus, self.device
                )
            case _:
                raise ValueError(f"Invalid criterion: {config.criterion}")
        """
        self.env: Environment = Environment(config.num_sentences_per_batch, config.max_num_steps, 1.0, device)

        # Initialize actor and critic networks
        self.actor_critic = ActorCritic(  # ALG STEP 1
            self.state_dim, config.embedding_dim, config.num_non_terminals, config.n_layer, config.n_head
        )
        self.actor_critic.to(device)

        # Initialize optimizers for actor_critic
        self.optim = Adam(self.actor_critic.parameters(), lr=config.lr)

        logger.info(
            f"PPO initialized with persistent_dir: {persistent_dir}, "
            f"device: {device}, config: {config}\n"
        )

        # Save the config as a JSON file in the persistent directory
        config_str: str = jsbeautifier.beautify(config.to_json())
        (persistent_dir / "config.json").write_text(config_str)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'time_ns': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'advantages': [],  # advantages in current iteration
            'pos_actor_losses': [],  # losses of actor network in current iteration
            'sym_actor_losses': [],  # losses of actor network in current iteration
            'critic_losses': [],  # losses of critic network in current iteration
            'pos_entropy': [],  # entropy of actor network in current iteration
            'sym_entropy': [],  # entropy of actor network in current iteration
        }

    def pos_tags_to_symbols(self, pos_tags: torch.Tensor) -> torch.Tensor:
        mask = pos_tags != -1
        symbols = pos_tags + mask * self.config.num_non_terminals
        return symbols
    
    def rule_mask_loss(self, rule_mask: torch.Tensor, sum_objective:float) -> torch.Tensor:
        """
        Calculate the loss for the rule mask.
        The aim is to limit the number RHS that an abstract symbol can produce to enforce clustering and compression.
        The sum of weights for each LHS should be as close to sum objective as possible.
        """
        # Calculate the sum of weights for each LHS
        lhs_weights = torch.sum(torch.exp(rule_mask), dim=(1,2)) # Sum weights over RHS. Exponential to go from logits to weights.
        # Calculate the loss as the mean squared error between lhs_weights and sum_objective
        loss = F.mse_loss(lhs_weights, sum_objective)
        return loss


    def learn(self, total_timesteps: int) -> None:
        """
        Train the actor and critic networks.
        Here is where the main PPO algorithm resides.

        :param total_timesteps: the total number of timesteps to train for
        """
        logger.info(
            f"Learning... Running {self.config.max_num_steps} timesteps per episode, "
            f"{self.config.num_sentences_per_batch} episodes per batch for a total of {total_timesteps} timesteps"
        )
        t_so_far: int = 0  # Timesteps simulated so far
        i_so_far: int = 0  # Iterations ran so far

        for epoch in range(self.config.num_epochs):
            for batch_t_stc, batch_t_spans in self.train_dataloader:  # ALG STEP 2
                if t_so_far >= total_timesteps:
                    break
                # ALG STEP 3, batch simulation
                # TODO
                #self.actor_critic.to(torch.device('cpu'))
                self.actor_critic.eval()
                with torch.no_grad():
                    
                    self.env.rollout(
                        self.actor_critic, self.pos_tags_to_symbols(batch_t_stc), batch_t_spans)
                    batch_obs, batch_rtgs, positions, positions_log_probs, symbols, symbols_log_probs, mask_position, mask_symbol, ep_rtgs = self.env.collect_data_batch(self.config.gamma)
                    # NOTE: batch_obs and batch_rtgs contain one extra timestep at the end
                    batch_lens: torch.Tensor = self.env.ep_len
                device = self.device
                self.actor_critic.to(device)
                self.actor_critic.train()


                # Increment timesteps so far and iterations so far
                t_so_far += sum(batch_lens)
                i_so_far += 1

                # Logging timesteps so far and iterations so far
                self.logger['t_so_far'] = t_so_far
                self.logger['i_so_far'] = i_so_far
                self.logger['batch_lens'] = batch_lens
                self.logger['batch_rews'] = ep_rtgs
                one_hot_positions = F.one_hot(positions.long(), num_classes=60).float()
                positions_histogram = one_hot_positions.sum(dim=0)
                one_hot_symbols = F.one_hot(symbols.long(), num_classes=self.config.num_non_terminals).float()
                symbols_histogram = one_hot_symbols.sum(dim=0)
                logger.info(f"positions histogram: {positions_histogram}")
                logger.info(f"symbols histogram: {symbols_histogram}")

                # Calculate advantage at k-th iteration
                with torch.no_grad():
                    if PURE_REINFORCE:
                        A_k: torch.Tensor = batch_rtgs
                    else:
                        V = self.actor_critic.state_val(batch_obs)
                        A_k: torch.Tensor = batch_rtgs - V.detach()  # ALG STEP 5

                # One of the few tricks I use that isn't in the pseudocode. Normalizing advantages
                # isn't theoretically necessary, but in practice it decreases the variance of
                # our advantages and makes convergence much more stable and faster. I added this because
                # solving some environments was too unstable without it.
                    #A_k: torch.Tensor = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                    P_A_k: torch.Tensor = A_k[mask_position]
                    S_A_k: torch.Tensor = A_k[mask_symbol]
                    self.logger['advantages'] = A_k

                n_updates_per_iteration = 1 if PURE_REINFORCE else self.config.n_updates_per_iteration
                # This is the loop where we update our network for some n epochs
                for _ in range(n_updates_per_iteration):  # ALG STEP 6 & 7
                    # Calculate V_phi and pi_theta(a_t | s_t)
                    curr_pos_log_probs, pos_dist_entropy, curr_sym_log_probs, sym_dist_entropy, _ = self.actor_critic.evaluate(batch_obs[mask_position], positions, symbols)
                    if PURE_REINFORCE:
                        pos_actor_loss = -(curr_pos_log_probs * P_A_k + self.config.entropy_weight*pos_dist_entropy).mean()
                        sym_actor_loss = -(curr_sym_log_probs * S_A_k + self.config.entropy_weight*sym_dist_entropy).mean()
                        critic_loss = torch.tensor([0.0])
                        loss = pos_actor_loss + sym_actor_loss

                    else:
                        V = self.actor_critic.state_val(batch_obs)

                        # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                        # NOTE: we just subtract the logs, which is the same as
                        # dividing the values and then canceling the log with e^log.
                        # For why, we use log probabilities instead of actual probabilities,
                        # here's a great explanation:
                        # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                        # TL;DR makes gradient ascent easier behind the scenes.
                        pos_ratios: torch.Tensor = torch.exp(curr_pos_log_probs - positions_log_probs)
                        sym_ratios: torch.Tensor = torch.exp(curr_sym_log_probs - symbols_log_probs)

                        # Calculate surrogate losses.
                        pos_surr1: torch.Tensor = pos_ratios * P_A_k
                        pos_surr2: torch.Tensor = torch.clamp(pos_ratios, 1 - self.config.clip, 1 + self.config.clip) * P_A_k
                        sym_surr1: torch.Tensor = sym_ratios * S_A_k
                        sym_surr2: torch.Tensor = torch.clamp(sym_ratios, 1 - self.config.clip, 1 + self.config.clip) * S_A_k

                        # Calculate actor and critic losses.
                        # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                        # the performance function, but Adam minimizes the loss. So minimizing the negative
                        # performance function maximizes it.
                        pos_actor_loss: torch.Tensor = -(torch.min(pos_surr1, pos_surr2) + self.config.entropy_weight * pos_dist_entropy).mean()
                        sym_actor_loss: torch.Tensor = -(torch.min(sym_surr1, sym_surr2) + 0* self.config.entropy_weight * sym_dist_entropy).mean()
                        critic_loss: torch.Tensor = F.mse_loss(V, batch_rtgs)
                        loss: torch.Tensor = self.config.actor_weight * (pos_actor_loss+sym_actor_loss)/2 + self.config.critic_weight * critic_loss
                        assert loss.isnan().sum().item() == 0, f"loss must not have NaN, got {loss}"

                    # Calculate gradients and perform backward propagation for actor_critic network
                    self.optim.zero_grad()
                    loss.backward()
                    # Clip gradients to prevent exploding gradients
                    if self.config.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.gradient_clip)
                    self.optim.step()

                    # Log actor and critic loss
                    self.logger['pos_actor_losses'].append(pos_actor_loss.item())
                    self.logger['sym_actor_losses'].append(sym_actor_loss.item())
                    self.logger['critic_losses'].append(critic_loss.item())
                    self.logger['pos_entropy'].append(pos_dist_entropy.mean().item())
                    self.logger['sym_entropy'].append(sym_dist_entropy.mean().item())

                # Print a summary of our training so far
                self._log_summary()

                # Save our model if it's time
                if i_so_far % self.config.save_freq == 0:
                    path: Path = self.persistent_dir / "torch" / f"actor_critic_{i_so_far}.pth"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        self.actor_critic.state_dict(),
                        str(path)
                    )

                if i_so_far % self.config.entropy_weight_decay_freq == 0:
                    self.config.entropy_weight = max(
                        self.config.entropy_weight * self.config.entropy_weight_decay,
                        self.config.entropy_weight_min
                    )

        path: Path = self.persistent_dir / "torch" / f"actor_critic_{i_so_far}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.actor_critic.state_dict(),
            str(path)
        )




    def _log_summary(self):
        delta_t: float = (time.time_ns() - self.logger['time_ns']) / 1e9
        self.logger['time_ns'] = time.time_ns()

        t_so_far: int = self.logger['t_so_far']
        i_so_far: int = self.logger['i_so_far']
        avg_ep_lens: float = mean(self.logger['batch_lens'])
        avg_ep_rews: float = mean([sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        min_ep_rews: float = min([sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        max_ep_rews: float = max([sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_advantages = mean(abs(self.logger['advantages']))
        avg_pos_actor_loss: float = mean(self.logger['pos_actor_losses'])
        avg_sym_actor_loss: float = mean(self.logger['sym_actor_losses'])
        acg_critic_loss: float = mean(self.logger['critic_losses'])
        avg_pos_entropy: float = mean(self.logger['pos_entropy'])
        avg_sym_entropy: float = mean(self.logger['sym_entropy'])

        logger.info(f"iter: {i_so_far}, timesteps so far: {t_so_far}, iteration took {delta_t:.2f} secs")
        logger.info(
            f"avg episodic len: {avg_ep_lens}, "
            f"avg episodic rewards: {avg_ep_rews:.5f}, min episodic rewards: {min_ep_rews:.5f}, "
            f"max episodic rewards: {max_ep_rews:.5f}"
            f", avg advantages: {avg_advantages:.5f}"
        )
        logger.info(f"avg position actor loss: {avg_pos_actor_loss:.5f}, avg symbol actor loss:{avg_sym_actor_loss:.5f}, avg critic loss: {acg_critic_loss:.5f}")
        logger.info(
            f"avg position entropy: {avg_pos_entropy:.5f}, "
            f"avg symbol entropy: {avg_sym_entropy:.5f}"
        )
        '''
        logger.info(
            f"pos actor loss: {self.logger['pos_actor_losses']}, "
            f"sym actor loss: {self.logger['sym_actor_losses']}, "
        )
        '''
        self.writer.log(
            {
                'delta_t': delta_t,
                'avg_ep_lens': avg_ep_lens,
                'avg_ep_rews': avg_ep_rews,
                'min_ep_rews': min_ep_rews,
                'max_ep_rews': max_ep_rews,
                'avg_abs_advantages': avg_advantages,
                'avg_pos_actor_loss': avg_pos_actor_loss,
                'avg_sym_actor_loss': avg_sym_actor_loss,
                'avg_critic_loss': acg_critic_loss,
                'avg_pos_entropy': avg_pos_entropy,
                'avg_sym_entropy': avg_sym_entropy,

            }, commit=True
        )

        self.result_saver.save(
                f"last", i_so_far,
                self.actor_critic, commit=True
            )
        logger.info("\n")

        # Reset batch-specific logging data
        self.logger['pos_actor_losses'] = []
        self.logger['sym_actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['pos_entropy'] = []
        self.logger['sym_entropy'] = []
