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
from grammar_env.criterion import CoverageCriterion, Criterion, F1Criterion, ProbabilityCriterion
from grammar_env.grammar.binary_grammar import BinaryGrammar, BinaryGrammarFactory
from grammar_env.grammar.unary_grammar import SupervisedUnaryGrammar, UnaryGrammar
from env import Environment
from grammar import CompleteBinGrammar
from grammar_env.grammar_env import GrammarEnv
from result_saver import ResultSaver
from writer import Writer

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
    num_non_terminals: int = 4  # Number of non-terminals
    max_productions: int = 400  # Maximum number of productions

    # Criterion parameters
    criterion: str = "f1"  # Criterion to use for training
    num_sentences_per_score: int = 256  # Number of sentences used to score per criterion
    num_sentences_per_batch: int = 256  # Number of sentences to process per batch

    # Algorithm parameters
    episodes_per_batch: int = 2  # Number of episodes to run per batch
    n_updates_per_iteration: int = 5  # Number of times to update actor/critic per iteration
    lr: float = 2e-4  # Learning rate of optimizer
    gamma: float = 0.99  # Discount factor to be applied when calculating Rewards-To-Go
    clip: float = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
    actor_weight: float = 1.  # Weight of the actor loss
    critic_weight: float = 0.5  # Weight of the critic loss
    entropy_weight: float = 0.05  # Weight of the entropy loss
    entropy_weight_decay: float = 0.9  # Decay of the entropy weight
    entropy_weight_min: float = 0.01  # Minimum entropy weight
    entropy_weight_decay_freq: int = 5  # How often to decay the entropy weight

    max_num_steps: int = 100  # Maximum number of steps to run in the environment

    # Miscellaneous parameters
    save_freq: int = 20  # How often we save in number of iterations
    seed: int = 0  # Sets the seed of our program, used for reproducibility of results
    min_ep_rews_threshold: float = 0.  # Minimum episodic rewards threshold to log the grammar

    # Network parameters
    hidden_dim: int = 512
    n_layer: int = 3


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

        torch.manual_seed(config.seed)

        self.unary_grammar: UnaryGrammar = SupervisedUnaryGrammar(self.train_corpus)
        self.binary_grammar_factory: BinaryGrammarFactory = BinaryGrammarFactory(
            config.num_non_terminals, self.unary_grammar.num_pt
        )
        self.unary_grammar.to(self.device)

        self.result_saver: ResultSaver = ResultSaver(
            self.persistent_dir, self.writer,
            self.train_corpus, self.valid_corpus, self.device,
            config.num_sentences_per_score, config.num_sentences_per_batch
        )

        # Setup environment
        match config.criterion:
            case "prob":
                self._criterion: Criterion = ProbabilityCriterion(
                    self.train_corpus, self.device, -200.,
                    config.num_sentences_per_score, config.num_sentences_per_batch
                )
            case "cov":
                self._criterion: Criterion = CoverageCriterion(
                    self.train_corpus, self.device,
                    config.num_sentences_per_score, config.num_sentences_per_batch
                )
            case "f1":
                self._criterion: Criterion = F1Criterion(
                    self.train_corpus, self.device,
                    config.num_sentences_per_score, config.num_sentences_per_batch
                )
            case _:
                raise ValueError(f"Invalid criterion: {config.criterion}")
        self.env: Environment = Environment(config.num_sentences_per_batch, config.max_num_steps, CompleteBinGrammar)
        self.state_dim = self.env.num_r
        self.action_dim = self.env.num_r

        # Initialize actor and critic networks
        self.actor_critic = ActorCritic(  # ALG STEP 1
            self.state_dim, self.action_dim, config.hidden_dim,
            config.n_layer
        )

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
            'actor_losses': [],  # losses of actor network in current iteration
            'critic_losses': [],  # losses of critic network in current iteration
        }

    def learn(self, total_timesteps: int) -> None:
        """
        Train the actor and critic networks.
        Here is where the main PPO algorithm resides.

        :param total_timesteps: the total number of timesteps to train for
        """
        logger.info(
            f"Learning... Running {self.config.max_productions} timesteps per episode, "
            f"{self.config.episodes_per_batch} episodes per batch for a total of {total_timesteps} timesteps"
        )
        t_so_far: int = 0  # Timesteps simulated so far
        i_so_far: int = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # ALG STEP 3, batch simulation
            # TODO
            self.actor_critic.to(torch.device('cpu'))
            self.actor_critic.eval()
            with torch.no_grad():
                self.rollout()
            device = self.device
            self.actor_critic.to(device)
            self.actor_critic.train()

            batch_obs: torch.Tensor = torch.from_numpy(self.env.state).to(device)
            batch_acts: torch.Tensor = torch.from_numpy(self.env.actions).to(device)
            batch_log_probs: torch.Tensor = torch.from_numpy(self.env.actions_log_probs).to(device)
            batch_rtgs: torch.Tensor = torch.from_numpy(self.env.compute_rtgs()).to(device)
            batch_lens: torch.Tensor = torch.from_numpy(self.env.ep_len).to(device)

            # Increment timesteps so far and iterations so far
            t_so_far += sum(batch_lens)
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            _, _, V = self.actor_critic.evaluate(batch_obs, batch_acts)
            A_k: torch.Tensor = batch_rtgs - V.detach()  # ALG STEP 5

            # One of the few tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k: torch.Tensor = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.config.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                curr_log_probs, dist_entropy, V = self.actor_critic.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why, we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios: torch.Tensor = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1: torch.Tensor = ratios * A_k
                surr2: torch.Tensor = torch.clamp(ratios, 1 - self.config.clip, 1 + self.config.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss: torch.Tensor = -(torch.min(surr1, surr2) + self.config.entropy_weight * dist_entropy).mean()
                critic_loss: torch.Tensor = F.mse_loss(V, batch_rtgs)
                loss: torch.Tensor = self.config.actor_weight * actor_loss + self.config.critic_weight * critic_loss
                assert loss.isnan().sum().item() == 0, f"loss must not have NaN, got {loss}"

                # Calculate gradients and perform backward propagation for actor_critic network
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Log actor and critic loss
                self.logger['actor_losses'].append(actor_loss.item())
                self.logger['critic_losses'].append(critic_loss.item())

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

    def rollout(self):
        """
        Collect batch of data from simulation.
        As PPO is an on-policy algorithm, we need to collect a fresh batch
        of data each time we iterate the actor/critic networks.

        :return: RolloutBuffer containing the batch data
        """
        # Batch data.

        time_s = time.time()


        # Sample sentences from the training corpus (sentences are padded to max length with index -1)
        batch_sentences = self.train_corpus.get_dataloader(self.config.num_sentences_per_batch)

        # Compute actual lengths (ignoring -1 values)
        seq_lengths = torch.tensor(batch_sentences!=-1).sum(axis=1)

        # Reset the environment.
        self.env.reset(seq_lengths.max().item())
        self.env.init_state(batch_sentences)

        # Run an episode for max_timesteps_per_episode timesteps
        ep_t: int = 0
        for ep_t in range(self.config.max_num_steps):

            # Calculate action and make a step in the env.
            current_states, not_done = self.env.get_state()
            action, log_prob, _ = self.actor_critic.act(torch.tensor(current_states[not_done]))
            assert action.shape == log_prob.shape == (len(not_done),1)
            padded_action = np.full(self.env.num_episodes, -1)
            padded_action[not_done] = action.squeeze().cpu().numpy()
            padded_log_prob = np.full(self.env.num_episodes, -1)
            padded_log_prob[not_done] = log_prob.squeeze().cpu().numpy()
            self.env.step(padded_action, padded_log_prob)

            # If the environment tells us the episode is terminated, break
            if self.env.stop():
                break



        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = self.env.compute_rtgs()
        self.logger['batch_lens'] = self.env.ep_len
        logger.info(f"Rollout done. {self.config.episodes_per_batch} episodes ran in {time.time() - time_s: .2f} secs")



    def _log_summary(self):
        delta_t: float = (time.time_ns() - self.logger['time_ns']) / 1e9
        self.logger['time_ns'] = time.time_ns()

        t_so_far: int = self.logger['t_so_far']
        i_so_far: int = self.logger['i_so_far']
        avg_ep_lens: float = mean(self.logger['batch_lens'])
        avg_ep_rews: float = mean([sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        min_ep_rews: float = min([sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        max_ep_rews: float = max([sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss: float = mean(self.logger['actor_losses'])
        acg_critic_loss: float = mean(self.logger['critic_losses'])

        logger.info(f"iter: {i_so_far}, timesteps so far: {t_so_far}, iteration took {delta_t:.2f} secs")
        logger.info(
            f"avg episodic len: {avg_ep_lens}, "
            f"avg episodic rewards: {avg_ep_rews:.5f}, min episodic rewards: {min_ep_rews:.5f}, "
            f"max episodic rewards: {max_ep_rews:.5f}"
        )
        logger.info(f"avg actor loss: {avg_actor_loss:.5f}, avg critic loss: {acg_critic_loss:.5f}")
        self.writer.log(
            {
                'delta_t': delta_t,
                'avg_ep_lens': avg_ep_lens,
                'avg_ep_rews': avg_ep_rews,
                'min_ep_rews': min_ep_rews,
                'max_ep_rews': max_ep_rews,
                'avg_actor_loss': avg_actor_loss,
                'avg_critic_loss': acg_critic_loss
            }, commit=False
        )

        if min_ep_rews >= self.config.min_ep_rews_threshold:
            binary_grammar: BinaryGrammar = self.binary_grammar_factory.create(
                self.env.state_to_reduced(self.env.state)
            )
            opt_binary_grammar: Optional[BinaryGrammar] = self._criterion.opt_binary_grammar

            if opt_binary_grammar is not None:
                self.result_saver.save(
                    f"opt", i_so_far,
                    opt_binary_grammar, self.unary_grammar, commit=False
                )

            self.result_saver.save(
                f"last", i_so_far,
                binary_grammar, self.unary_grammar, commit=True
            )
        logger.info("\n")

        # Reset batch-specific logging data
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
