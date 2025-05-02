import torch
from typing import Optional
from logging import getLogger
from actor_critic import ActorCritic
import time

logger = getLogger(__name__)


def min_padding(x: torch.Tensor):
    # x is a 2D tensor containing a batch of sentences with -1 as padding

    # Create a mask where padding is False
    mask = x != -1

    # Find the last non-padding index per sentence
    lengths = mask.int().sum(dim=1)  # Number of non-padding tokens per sentence

    # Get the maximum of these lengths
    max_len = lengths.max().item()

    # Optionally truncate the tensor to that max_len
    x_trimmed = x[:, :max_len]

    return x_trimmed, lengths

def fuse_and_shift(arr: torch.Tensor, idx: torch.Tensor, value:torch.Tensor, padding_value):
    assert torch.all((idx >= 0) & (idx < arr.shape[1] - 1)), "Index out of bounds"
    
    arr_copy = arr.clone()
    value = value.to(arr_copy.dtype)
    # Insert fusion value
    arr_copy[:,idx] = value
    
    # Shift elements from i+2 to end
    for i in range(arr.shape[0]):
        # Shift elements in the row
        arr_copy[i, idx[i]+1:-1] = arr[i, idx[i]+2:]
    #arr_np[:,idx_np+1:-1] = arr_np[:,idx_np+2:]
    
    # Pad the last element
    arr_copy[:, -1] = padding_value

    return arr_copy

def fuse_spans_and_shift(arr: torch.Tensor, idx: torch.Tensor, padding_value):
    assert torch.all((idx >= 0) & (idx < arr.shape[1] - 1)), "Index out of bounds"
    assert idx.shape[0] == arr.shape[0], "Index and array must have the same number of rows"
    
    arr_copy = arr.clone()

    # Fuse spans
    fused_spans = torch.stack([arr_copy[torch.arange(arr.shape[0]), idx, 0], arr_copy[torch.arange(arr.shape[0]), idx + 1, 1]], dim=1)
    arr_copy[:,idx, :] = fused_spans
    
    # Shift elements from i+2 to end
    for i in range(arr.shape[0]):
        # Shift elements in the row
        arr_copy[i, idx[i]+1:-1] = arr[i, idx[i]+2:]
    
    # Pad the last element
    arr_copy[:, -1, :] = padding_value

    return arr_copy, fused_spans



class Environment:
    def __init__(self, num_episodes, max_num_steps, success_weight:float, device: torch.device):
        self.device = device
        self.max_num_steps = max_num_steps
        self.num_episodes = num_episodes
        self.success_weight = success_weight

        self.state : Optional[torch.Tensor] = None
        self.spans_sentences: Optional[torch.Tensor] = None
        self.spans_lists: list[list[tuple[int, int]]] = [[] for _ in range(num_episodes)]
        self.sentence_lengths: torch.Tensor = torch.zeros(num_episodes, dtype=torch.int32, device=device)

        self.positions : torch.Tensor = torch.full((num_episodes, max_num_steps), -1, dtype=torch.int32, device=device)
        self.positions_log_probs : torch.Tensor = torch.full((num_episodes, max_num_steps), float('-inf'), dtype=torch.float32, device=device)
        self.symbols : torch.Tensor = torch.full((num_episodes, max_num_steps), -1, dtype=torch.int32, device=device)
        self.symbols_log_probs : torch.Tensor = torch.full((num_episodes, max_num_steps), float('-inf'), dtype=torch.float32, device=device)
        self.rew : torch.Tensor = torch.zeros((num_episodes, max_num_steps), dtype=torch.bool, device=device)
        self.rhs_index : torch.Tensor = torch.full((num_episodes, max_num_steps), -1, dtype=torch.int32, device=device)

        self.step_count : int = 0
        self.done : torch.Tensor = torch.zeros(num_episodes, dtype=torch.bool, device=device)
        self.ep_len : torch.Tensor = torch.full((num_episodes,), max_num_steps, dtype=torch.int32, device=device)

    def reset(self, batch_sentences: torch.Tensor, batch_spans: list[list[tuple[int, int]]]):
        # batch_sentences: [num_episodes, sentence_length]
        self.state = torch.full(
            (self.num_episodes, self.max_num_steps + 1, batch_sentences.size(1)),
            -1,
            dtype=torch.int32,
            device=self.device
        )
        self.gt_spans = [set(span) for span in batch_spans]
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

        self.positions.fill_(-1)
        self.positions_log_probs.fill_(float('-inf'))
        self.symbols.fill_(-1)
        self.symbols_log_probs.fill_(float('-inf'))
        self.rew.zero_()
        self.step_count = 0
        self.done.zero_()
        self.ep_len.fill_(self.max_num_steps)
        self.check_done()


    def stop(self):
        return self.step_count >= self.max_num_steps - 1 or self.done.all()

    def check_done(self):
        curr_state = self.state[:, self.step_count, :]
        mask = (curr_state[:, 1:] == -1).all(dim=1) # & (curr_state[:, 0] == 0) # Here done does not mean successfulß
        self.done = mask.to(self.device)
        self.ep_len[mask] = torch.minimum(self.ep_len[mask], torch.tensor(self.step_count, device=self.device))

    def step(self, positions: torch.Tensor, positions_log_probs: torch.Tensor, symbols: torch.Tensor, symbols_log_probs: torch.Tensor):
        not_done = ~self.done
        assert positions.shape == positions_log_probs.shape == symbols.shape == symbols_log_probs.shape == (not_done.sum().item(),), f"Shape mismatch: {positions.shape}, {positions_log_probs.shape}, {symbols.shape}, {symbols_log_probs.shape}, {(not_done.sum().item(),)}"

        current_state = self.state[not_done, self.step_count, :]
        current_spans = self.spans_sentences[not_done]

        next_sentence = fuse_and_shift(current_state, positions, symbols, -1)
        next_spans, new_spans = fuse_spans_and_shift(current_spans, positions, -1)

        self.state[:, self.step_count + 1, :] = self.state[:, self.step_count, :]
        self.state[not_done, self.step_count + 1, :] = next_sentence
        self.spans_sentences[not_done, :, :] = next_spans

        not_done_idx = torch.arange(self.num_episodes, device=self.device)[not_done]
        for i in range(len(new_spans)):
            self.spans_lists[not_done_idx[i]].append((new_spans[i, 0].item(), new_spans[i, 1].item()))

        self.positions[not_done, self.step_count] = positions
        self.positions_log_probs[not_done, self.step_count] = positions_log_probs
        self.symbols[not_done, self.step_count] = symbols
        self.symbols_log_probs[not_done, self.step_count] = symbols_log_probs

        self.rew[not_done, self.step_count] = self.supervised_spans_reward(not_done, new_spans)


        self.step_count += 1
        self.check_done()

    def get_state(self):
        not_done = ~self.done
        current_state, _ = min_padding(self.state[not_done, self.step_count, :])
        return current_state, not_done

    def get_rewards(self):
        return self.rew[:, :self.step_count].clone()

    def compute_rtgs(self, gamma: float) -> torch.Tensor:
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.
        """
        batch_rtgs = torch.zeros((self.num_episodes, self.ep_len.max()+1), dtype=torch.float32, device=self.device)

        discounted_reward = self.success_weight * self.success()
        for i in range(self.num_episodes):
            batch_rtgs[i, self.ep_len[i]] = discounted_reward[i]
            for t in reversed(range(self.ep_len[i])):
                discounted_reward[i] = self.rew[i, t] + gamma * discounted_reward[i]
                batch_rtgs[i, t] = discounted_reward[i]
        return batch_rtgs
    
    def supervised_spans_reward(self, not_done: torch.Tensor, new_spans: torch.Tensor):
        """
        Compute the reward for the new spans based on the ground truth spans.
        """
        gt_spans = [span for span, nd in zip(self.gt_spans, not_done) if nd]
        reward = torch.zeros(len(new_spans), dtype=torch.bool, device=self.device)

        for i in range(len(new_spans)):
            if (new_spans[i,0].item(), new_spans[i,1].item()) in gt_spans[i]:
                reward[i] = 1
        return reward
    
    def success(self) -> torch.Tensor:
        return self.done & (self.state[:, self.step_count, 0] == 0)

    
    def collect_data_batch(self, gamma: float):
        self.check_done()
        batch_states = []
        batch_positions = []
        batch_positions_log_probs = []
        batch_rewards = []
        batch_symbols = []
        batch_symbols_log_probs = []
        mask_position = []
        mask_symbol = []
        rews_to_go = self.compute_rtgs(gamma)
        for i in range(self.num_episodes):
            batch_states += self.state[i, :self.ep_len[i]+1, :].tolist()
            batch_positions += self.positions[i, :self.ep_len[i]].tolist()
            batch_positions_log_probs += self.positions_log_probs[i, :self.ep_len[i]].tolist()
            batch_rewards += (rews_to_go[i, :self.ep_len[i]+1]).tolist()
            batch_symbols += self.symbols[i, :self.ep_len[i]].tolist()
            batch_symbols_log_probs += self.symbols_log_probs[i, :self.ep_len[i]].tolist()
            mask_position += [1] * self.ep_len[i] + [0]
            mask_symbol += [0] + [1] * self.ep_len[i]
        return torch.tensor(batch_states, dtype=torch.int32, device=self.device), \
               torch.tensor(batch_rewards, dtype=torch.float32, device=self.device), \
                torch.tensor(batch_positions, dtype=torch.int32, device=self.device), \
                torch.tensor(batch_positions_log_probs, dtype=torch.float32, device=self.device), \
                torch.tensor(batch_symbols, dtype=torch.int32, device=self.device), \
                torch.tensor(batch_symbols_log_probs, dtype=torch.float32, device=self.device), \
                torch.tensor(mask_position, dtype=torch.bool, device=self.device), \
                torch.tensor(mask_symbol, dtype=torch.bool, device=self.device), \
               rews_to_go
    

    def rollout(self, actor_critic: ActorCritic, batch_sentences: torch.Tensor, batch_spans: list[list[tuple[int, int]]], evaluate: bool = False):
        """
        Collect batch of data from simulation.
        As PPO is an on-policy algorithm, we need to collect a fresh batch
        of data each time we iterate the actor/critic networks.

        :return: RolloutBuffer containing the batch data
        """
        time_s = time.time()

        assert batch_sentences.shape[0] == len(batch_spans) == self.num_episodes, f"Data batch size {batch_sentences.shape[0]} does not match num_episodes {self.num_episodes}"

        actor_critic.eval()

        batch_sentences = batch_sentences.to(self.device)

        # Reset the environment.
        self.reset(batch_sentences, batch_spans)

        # Run an episode for max_timesteps_per_episode timesteps
        ep_t: int = 0
        for ep_t in range(self.max_num_steps):

            current_states, not_done = self.get_state()
            position, position_log_prob, symbol, symbol_log_prob, _ = actor_critic.act(current_states, max_prob=evaluate)
            assert position.shape == position_log_prob.shape == symbol.shape == symbol_log_prob.shape == (not_done.sum().item(),), f"Shape mismatch: {position.shape}, {position_log_prob.shape}, {symbol.shape}, {symbol_log_prob.shape}, {(not_done.sum().item(),)}"

            self.step(position, position_log_prob, symbol, symbol_log_prob)

            if self.stop():
                break
        
        logger.info(f"Rollout done. {self.num_episodes} episodes ran in {time.time() - time_s: .2f} secs")

