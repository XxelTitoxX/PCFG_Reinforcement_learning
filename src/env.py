import torch
from grammar import CompleteBinGrammar
from typing import Optional
from grammar_env.criterion.criterion import Criterion
from grammar_env.corpus.corpus import Corpus
from logging import getLogger
from actor_critic import ActorCritic
import time

logger = getLogger(__name__)




class Environment:
    def __init__(self, num_episodes, max_num_steps, bin_grammar: CompleteBinGrammar, criterion:Criterion, success_weight:float, device: torch.device):
        self.device = device
        self.bin_grammar = bin_grammar
        self.max_num_steps = max_num_steps
        self.num_episodes = num_episodes
        self.criterion = criterion # WARNING : criterion is related to a corpus
        self.success_weight = success_weight

        self.state : Optional[torch.Tensor] = None
        self.sentence_idx : Optional[torch.Tensor] = None
        self.spans_sentences: Optional[torch.Tensor] = None
        self.spans_lists: list[list[tuple[int, int]]] = [[] for _ in range(num_episodes)]
        self.sentence_lengths: torch.Tensor = torch.zeros(num_episodes, dtype=torch.int32, device=device)

        self.actions : torch.Tensor = torch.full((num_episodes, max_num_steps), -1, dtype=torch.int32, device=device)
        self.actions_log_probs : torch.Tensor = torch.zeros((num_episodes, max_num_steps), dtype=torch.float32, device=device)
        self.rew : torch.Tensor = torch.zeros((num_episodes, max_num_steps), dtype=torch.bool, device=device)
        self.rhs_index : torch.Tensor = torch.full((num_episodes, max_num_steps), -1, dtype=torch.int32, device=device)

        self.step_count : int = 0
        self.done : torch.Tensor = torch.zeros(num_episodes, dtype=torch.bool, device=device)
        self.ep_len : torch.Tensor = torch.full((num_episodes,), max_num_steps, dtype=torch.int32, device=device)

    def reset(self, batch_sentences: torch.Tensor, batch_indices: torch.Tensor):
        # batch_sentences: [num_episodes, sentence_length]
        self.state = torch.full(
            (self.num_episodes, self.max_num_steps + 1, batch_sentences.size(1)),
            -1,
            dtype=torch.int32,
            device=self.device
        )
        self.sentence_idx = batch_indices.to(self.device)
        self.state[:, 0, :] = batch_sentences.to(self.device)
        self.sentence_lengths = batch_sentences.ne(-1).sum(dim=1).to(self.device)

        self.spans_sentences = torch.full(
            (self.num_episodes, batch_sentences.size(1), 2),
            -1,
            dtype=torch.int32,
            device=self.device
        )

        valid_symbols = batch_sentences != -1
        start_spans = torch.arange(batch_sentences.size(1), device=self.device, dtype=torch.int32)
        all_start_spans = torch.stack([start_spans, start_spans], dim=1)  # [length, 2]
        all_start_spans = all_start_spans.unsqueeze(0).repeat(self.num_episodes, 1, 1)
        self.spans_sentences[valid_symbols] = all_start_spans[valid_symbols]

        self.spans_lists = [[] for _ in range(self.num_episodes)]

        self.actions.fill_(-1)
        self.rhs_index.fill_(-1)
        self.rew.zero_()
        self.step_count = 0
        self.done.zero_()
        self.ep_len.fill_(self.max_num_steps)
        self.actions_log_probs.zero_()
        self.check_done()


    def stop(self):
        return self.step_count >= self.max_num_steps - 1 or self.done.all()

    def check_done(self):
        curr_state = self.state[:, self.step_count, :]
        mask = (curr_state[:, 1:] == -1).all(dim=1) & (curr_state[:, 0] == 0)
        self.done = mask.to(self.device)
        self.ep_len[mask] = torch.minimum(self.ep_len[mask], torch.tensor(self.step_count, device=self.device))

    def step(self, actions: torch.Tensor, actions_log_probs: torch.Tensor):
        not_done = ~self.done

        current_state = self.state[not_done, self.step_count, :]
        current_actions = actions[not_done]

        new_sentence, valid_parsing, rhs_index = self.bin_grammar.parsing_step(current_state, current_actions)

        self.state[:, self.step_count + 1, :] = self.state[:, self.step_count, :]
        self.state[not_done, self.step_count + 1, :] = new_sentence

        self.actions[:, self.step_count] = actions
        self.actions_log_probs[:, self.step_count] = actions_log_probs

        self.rhs_index[not_done, self.step_count] = rhs_index
        self.rew[not_done, self.step_count] = valid_parsing

        # Compute spans
        spans_sentences, new_spans = self.bin_grammar.span_computation_step(
            self.rhs_index[:, self.step_count], self.spans_sentences
        )
        self.spans_sentences = spans_sentences

        for i in range(self.num_episodes):
            if new_spans[i, 0] != -1:
                self.spans_lists[i].append((new_spans[i, 0].item(), new_spans[i, 1].item()))

        self.step_count += 1
        self.check_done()

    def get_state(self):
        not_done = ~self.done
        return self.state[:, self.step_count, :].clone(), not_done

    def get_rewards(self):
        return self.rew[:, :self.step_count].clone()

    def compute_rtgs(self, gamma: float) -> torch.Tensor:
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.
        """
        batch_rtgs = torch.zeros_like(self.rew, dtype=torch.float32, device=self.device)

        discounted_reward = self.success_weight * self.success_reward()
        for i in range(self.num_episodes):
            for t in reversed(range(self.ep_len[i])):
                discounted_reward[i] = self.rew[i, t] + gamma * discounted_reward[i]
                batch_rtgs[i, t] = discounted_reward[i]
        print(f"Batch RTGs: {batch_rtgs}")
        return batch_rtgs
    
    def success_reward(self):
        return self.criterion.score_sentences(self)
    
    def collect_data_batch(self, gamma: float):
        self.check_done()
        batch_states = []
        batch_actions = []
        batch_actions_log_probs = []
        batch_rewards = []
        rews_to_go = self.compute_rtgs(gamma)
        for i in range(self.num_episodes):
            batch_states += self.state[i, :self.ep_len[i]-1, :].tolist()
            batch_actions += self.actions[i, :self.ep_len[i]-1].tolist()
            batch_actions_log_probs += self.actions_log_probs[i, :self.ep_len[i]-1].tolist()
            batch_rewards += rews_to_go[i, :self.ep_len[i]-1].tolist()
        assert torch.all(torch.tensor(batch_actions != -1)), "Batch of actions should only contain valid actions"
        return torch.tensor(batch_states, dtype=torch.int32, device=self.device), \
               torch.tensor(batch_actions, dtype=torch.int32, device=self.device), \
               torch.tensor(batch_actions_log_probs, dtype=torch.float32, device=self.device), \
               torch.tensor(batch_rewards, dtype=torch.float32, device=self.device), \
               rews_to_go
    

    def rollout(self, corpus : Corpus, actor_critic: ActorCritic, num_sentences_per_batch: int, max_num_steps: int):
        """
        Collect batch of data from simulation.
        As PPO is an on-policy algorithm, we need to collect a fresh batch
        of data each time we iterate the actor/critic networks.

        :return: RolloutBuffer containing the batch data
        """
        time_s = time.time()

        actor_critic.eval()

        # Sample sentences from the training corpus
        batch_sentences, batch_indices = next(iter(corpus.get_dataloader(num_sentences_per_batch)))
        batch_sentences = batch_sentences.to(self.device)
        batch_indices = batch_indices.to(self.device)

        # Reset the environment.
        self.reset(batch_sentences, batch_indices)

        # Run an episode for max_timesteps_per_episode timesteps
        ep_t: int = 0
        for ep_t in range(max_num_steps):

            current_states, not_done = self.get_state()
            action, log_prob, _ = actor_critic.act(current_states[not_done])
            assert action.shape == log_prob.shape == (not_done.sum(),)

            # Create padded tensors for actions and log_probs, filled with -1
            padded_action = torch.full((self.num_episodes,), -1, dtype=torch.long, device=self.device)
            padded_action[not_done] = action

            padded_log_prob = torch.full((self.num_episodes,), float('-inf'), dtype=torch.float, device=self.device)
            padded_log_prob[not_done] = log_prob

            self.step(padded_action,padded_log_prob)

            if self.stop():
                break
        
        #self.criterion.update_optimal_model(actor_critic)
        logger.info(f"Rollout done. {self.num_episodes} episodes ran in {time.time() - time_s: .2f} secs")

