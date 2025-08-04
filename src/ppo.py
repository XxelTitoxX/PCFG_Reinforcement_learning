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
from torch.distributions import Categorical

from actor_critic import ActorCritic
from grammar_env.corpus.corpus import Corpus
from env import Environment
from result_saver import ResultSaver
from writer import Writer
from grammar_env.criterion import F1Criterion
from n_gram import NGram

logger = getLogger(__name__)

# https://discuss.pytorch.org/t/cpu-usage-far-too-high-and-training-inefficient/57228/3
# https://discuss.pytorch.org/t/pytorch-v2-high-cpu-consumption/205990
# OpenMP uses all the cpu cores by default, leading to inefficiency.
torch.set_num_threads(1)


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
    num_non_terminals: int = 26  # Number of non-terminals

    # Criterion parameters
    criterion: str = "f1"  # Criterion to use for training
    num_sentences_per_score: int = 128  # Number of sentences used to score per criterion
    num_sentences_per_batch: int = 64  # Number of sentences to process per batch
    num_epochs: int = 10

    # Algorithm parameters
    n_updates_per_iteration: int = 10  # Number of times to update actor/critic per iteration
    lr: float = 2e-4  # Learning rate of optimizer
    gamma: float = 0.0  # Discount factor to be applied when calculating Rewards-To-Go
    clip: float = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
    actor_weight: float = 1.  # Weight of the actor loss
    critic_weight: float = 0.5  # Weight of the critic loss
    entropy_weight: float = 0.01  # Weight of the entropy loss
    entropy_weight_decay: float = 0.98  # Decay of the entropy weight
    entropy_weight_min: float = 0.01  # Minimum entropy weight
    entropy_weight_decay_freq: int = 10  # How often to decay the entropy weight

    max_num_steps: int = 70  # Maximum number of steps to run in the environment

    # Miscellaneous parameters
    save_freq: int = 20  # How often we save in number of iterations
    eval_freq: int = 10  # How often we evaluate the model in number of training iterations
    seed: int = 0  # Sets the seed of our program, used for reproducibility of results
    min_ep_rews_threshold: float = 0.  # Minimum episodic rewards threshold to log the grammar

    # Network parameters
    n_layer: int = 2
    n_head: int = 2
    embedding_dim: int = 64

    gradient_clip : Optional[float] = None
    pure_reinforce : bool = True


class PPO:
    def __init__(
            self, train_corpus: Corpus, valid_corpus: Corpus, persistent_dir: Path,
            writer: Writer, device: torch.device,
            config: PPOConfig, n_gram: Optional[NGram] = None
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
        self.train_iterator = iter(self.train_dataloader)
        self.valid_dataloader = valid_corpus.get_dataloader(config.num_sentences_per_score)

        self.result_saver: ResultSaver = ResultSaver(
            self.persistent_dir, self.writer,
            self.train_corpus, self.valid_corpus, self.device,
            config.num_sentences_per_score, config.max_num_steps,
        )

        # Setup environment
        self.env: Environment = Environment(config.max_num_steps, 0.0, device, n_gram=n_gram, symbol_freq=train_corpus.symbol_freq)

        # Initialize actor and critic networks
        self.actor_critic = ActorCritic(  # ALG STEP 1
            self.state_dim, config.embedding_dim, config.num_non_terminals, config.n_layer, config.n_head, train_corpus.vocab_size
        )
        self.actor_critic.to(device)

        self.control_f1_criterion = F1Criterion(device)

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
            'batch_returns': [],  # episodic returns in batch
            'advantages': [],  # advantages in current iteration
            'pos_actor_losses': [],  # losses of actor network in current iteration
            'sym_actor_losses': [],  # losses of actor network in current iteration
            'critic_losses': [],  # losses of critic network in current iteration
            'pos_entropy': [],  # entropy of actor network in current iteration
            'sym_entropy': [],  # entropy of actor network in current iteration
            'positions_histogram' : [],
            'symbols_histogram' : [],
            'position_rews': [],  # position rewards in current iteration
            'symbol_rews': [],  # symbol rewards in current iteration
        }



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

        epoch: int = 0
        while epoch < self.config.num_epochs and t_so_far < total_timesteps:  
            # ALG STEP 2, batch selection
            try:
                batch_t_stc = next(self.train_iterator)
                if len(batch_t_stc) != self.config.num_sentences_per_batch:
                    raise StopIteration
            except StopIteration:
                logger.info(f"EPOCH {epoch} finished, t_so_far: {t_so_far}, i_so_far: {i_so_far}")
                epoch += 1
                self.train_iterator = iter(self.train_dataloader)
                batch_t_stc = next(self.train_iterator)
            # ALG STEP 3, batch simulation
            self.actor_critic.eval()
            with torch.no_grad():
                
                self.env.rollout(
                    self.actor_critic, batch_t_stc, supervised_update=False)
                V, positions, positions_log_probs, pos_rtgs, symbols, symbols_log_probs, sym_rtgs, ep_rtgs, mask_before, mask_after = self.env.collect_data_batch(self.config.gamma) # mask_position, mask_symbol
                # NOTE: V contains one extra timestep at the end
                batch_lens: torch.Tensor = self.env.ep_len
                valid_timesteps: torch.Tensor = torch.arange(self.env.max_num_steps, device=self.device)[None, :] < batch_lens[:, None]
                self.logger['position_rews'] = self.env.rew.float().sum(dim=0) / (valid_timesteps.sum(dim=0) + 1e-9)
                self.logger['symbol_rews'] = self.env.sym_rew.float().sum(dim=0) / (valid_timesteps.sum(dim=0) + 1e-9)
                print(f"Average span reward : {torch.mean(self.env.rew.sum(dim=1)/(batch_lens.float()+1e-9))}")
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
            self.logger['batch_returns'] = ep_rtgs
            one_hot_positions = F.one_hot(positions.long(), num_classes=60).float()
            positions_histogram = one_hot_positions.sum(dim=0)
            one_hot_symbols = F.one_hot(symbols.long(), num_classes=self.config.num_non_terminals).float()
            symbols_histogram = one_hot_symbols.sum(dim=0)
            self.logger['positions_histogram'] = positions_histogram
            self.logger['symbols_histogram'] = symbols_histogram

            # Calculate advantage at k-th iteration
            with torch.no_grad():
                if self.config.pure_reinforce:
                    #P_A_k: torch.Tensor = (pos_rtgs - pos_rtgs.mean()) / (pos_rtgs.std() + 1e-10)  # Batch normalization
                    #S_A_k: torch.Tensor = (sym_rtgs - sym_rtgs.mean()) / (sym_rtgs.std() + 1e-10)  # Batch normalization
                    P_A_k: torch.Tensor = pos_rtgs  # No normalization
                    S_A_k: torch.Tensor = sym_rtgs  # No normalization
                else:
                    A_k: torch.Tensor = pos_rtgs+sym_rtgs - V[mask_before]  # ALG STEP 5
                    A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
                    P_A_k: torch.Tensor = A_k  # Position advantages
                    S_A_k: torch.Tensor = A_k  # Symbol advantages

            # One of the few tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
                self.logger['advantages'] = P_A_k

            n_updates_per_iteration = 1 if self.config.pure_reinforce else self.config.n_updates_per_iteration
            # This is the loop where we update our network for some n epochs
            for idx_upd in range(n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                curr_pos_log_probs, pos_dist_entropy, curr_sym_log_probs, sym_dist_entropy, V, add_obj = self.env.replay(self.actor_critic, b_add_obj=False) # [mask_position]
                if self.config.pure_reinforce:
                    alternator = idx_upd % 2
                    pos_actor_loss = -(curr_pos_log_probs * P_A_k + self.config.entropy_weight*pos_dist_entropy).mean()
                    sym_actor_loss = -(curr_sym_log_probs * S_A_k + self.config.entropy_weight*sym_dist_entropy).mean()
                    critic_loss = torch.tensor([0.0])
                    #loss = (1-alternator)*pos_actor_loss + alternator*sym_actor_loss - 0.5*add_obj
                    loss = pos_actor_loss + sym_actor_loss - 0.5*add_obj
                    logger.info(f"Additional objective: {add_obj.item()}")

                else:

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
                    sym_actor_loss: torch.Tensor = -(torch.min(sym_surr1, sym_surr2) + self.config.entropy_weight * sym_dist_entropy).mean()
                    critic_loss: torch.Tensor = F.mse_loss(V, pos_rtgs+sym_rtgs)
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
            if i_so_far % self.config.eval_freq == 0:
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
        position_rews: torch.Tensor = self.logger['position_rews']
        symbol_rews: torch.Tensor = self.logger['symbol_rews']
        avg_ep_rews: float = mean([sum(ep_rews) for ep_rews in self.logger['batch_returns']])
        min_ep_rews: float = min([sum(ep_rews) for ep_rews in self.logger['batch_returns']])
        max_ep_rews: float = max([sum(ep_rews) for ep_rews in self.logger['batch_returns']])
        avg_advantages = mean(abs(self.logger['advantages']))
        avg_pos_actor_loss: float = mean(self.logger['pos_actor_losses'])
        avg_sym_actor_loss: float = mean(self.logger['sym_actor_losses'])
        acg_critic_loss: float = mean(self.logger['critic_losses'])
        avg_pos_entropy: float = mean(self.logger['pos_entropy'])
        avg_sym_entropy: float = mean(self.logger['sym_entropy'])
        pos_histogram = self.logger['positions_histogram']
        sym_histogram = self.logger['symbols_histogram']
        global_pos_entropy = Categorical(pos_histogram/pos_histogram.sum()).entropy().mean().item()
        global_sym_entropy = Categorical(sym_histogram/sym_histogram.sum()).entropy().mean().item()
        f1_score = torch.mean(self.control_f1_criterion.score_sentences(self.env)).item()

        logger.info("*" * 20)
        logger.info(f"ITER: {i_so_far}, timesteps so far: {t_so_far}, iteration took {delta_t:.2f} secs")
        logger.info(
            f"positions histogram: {pos_histogram}, "
        )
        logger.info(
            f"symbols histogram: {sym_histogram}, "
        )
        logger.info(f"Position rewards: {position_rews}, Symbol rewards: {symbol_rews}")
        logger.info(f"F1 score on train batch: {f1_score:.4f}")
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
                'pos_actor_avg_entropy': avg_pos_entropy,
                'sym_actor_avg_entropy': avg_sym_entropy,
                'simulation_pos_entropy': global_pos_entropy,
                'simulation_sym_entropy': global_sym_entropy,

            }, commit=False
        )

        self.result_saver.save(
                f"last", i_so_far,
                self.actor_critic,commit=True
            )
        logger.info("\n")

        # Reset batch-specific logging data
        self.logger['pos_actor_losses'] = []
        self.logger['sym_actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['pos_entropy'] = []
        self.logger['sym_entropy'] = []
