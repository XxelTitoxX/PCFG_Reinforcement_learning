import torch
from typing import Optional
from logging import getLogger
from actor_critic import ActorCritic
from grammar_env.corpus.sentence import GoldSpan
import time

logger = getLogger(__name__)

IDX_TO_NT_TAG = {
    0: 'X', 1: 'ADJP', 2: 'ADVP', 3: 'CONJP', 4: 'FRAG', 5: 'INTJ', 6: 'LST', 7: 'NAC', 8: 'NP',
    9: 'NX', 10: 'PP', 11: 'PRN', 12: 'PRT', 13: 'QP', 14: 'RRC', 15: 'S', 16: 'SBAR', 17: 'SBARQ',
    18: 'SINV', 19: 'SQ', 20: 'UCP', 21: 'VP', 22: 'WHADJP', 23: 'WHADVP', 24: 'WHNP', 25: 'WHPP'
}


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

def fuse_and_shift(arr: torch.Tensor, idx: torch.Tensor, value: torch.Tensor, padding_value):
    """
    For each row i in 'arr', fuses the values at position 'idx[i]' and 'idx[i]+1' into 'value[i]' and shifts the rest of the elements to the left.
    The last element of each row is set to 'padding_value'.
    """
    assert torch.all((idx >= 0) & (idx < arr.shape[1] - 1)), "Index out of bounds"
    
    arr_copy = arr.clone()
    value = value.to(arr_copy.dtype)
    
    for i in range(arr.shape[0]):
        fusion_index = idx[i].item()
        # Insert fusion value at position idx[i]
        arr_copy[i, fusion_index] = value[i]
        # Shift left the elements after idx[i]+1
        arr_copy[i, fusion_index+1:-1] = arr[i, fusion_index+2:]
        # Pad last element
        arr_copy[i, -1] = padding_value
    
    return arr_copy

def test_fuse_and_shift():
    arr = torch.tensor([[31, 31, 31, 31, 31, 26, 27, 30, 31, 35, 27, 31, 29, 28, 31, 26, 33, 35,                                                                                                     
         29, 28, 31, 37, 35, 26, 33, 33, 35, 28, 27, 27, 31, 31, 36, 35, 28, 31,                                                                                                                    
         29, 28, 31, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,                                                                                                                    
         -1, -1, -1],                                                                                                                                                                               
        [29, 31, 31, 31, 31, 31, 31, 31, 31, 35, 29, 27, 33, 27, 31, -1, -1, -1,                                                                                                                    
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,                                                                                                                    
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,                                                                                                                    
         -1, -1, -1],                                                                                                                                                                               
        [32, 35, 33, 30, 31, 29, 32, 31, 31, 35, 31, 31, 28, 31, 29, 28, 31, 31,                                                                                                                    
         31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,                                                                                                                    
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,                                                                                                                    
         -1, -1, -1]], dtype=torch.int32)
    idx = torch.tensor([3,5,2], dtype=torch.int32)
    value = torch.tensor([8,8,1], dtype=torch.int32)
    padding_value = -1
    expected_result = torch.tensor([[ 8,  8,  8,  8, 26, 27, 30, 31, 35, 27, 31, 29, 28, 31, 26, 33, 35, 29,                                                                                                     
         28, 31, 37, 35, 26, 33, 33, 35, 28, 27, 27, 31, 31, 36, 35, 28, 31, 29,                                                                                                                    
         28, 31, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,                                                                                                                    
         -1, -1, -1],                                                                                                                                                                               
        [ 8,  8,  8,  8,  8,  8, 31, 31, 35, 29, 27, 33, 27, 31, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1],
        [ 8,  8,  1, 31, 29, 32, 31, 31, 35, 31, 31, 28, 31, 29, 28, 31, 31, 31,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1]], dtype=torch.int32)
    result = fuse_and_shift(arr, idx, value, padding_value)
    print(f"Expected {expected_result}, but got {result}")
    

def fuse_spans_and_shift(arr: torch.Tensor, idx: torch.Tensor, padding_value):
    """
    For each row i in 'arr', fuses the spans at position 'idx[i]' and 'idx[i]+1' into a new span 
    taking the start of the first and end of the second. Then shifts the rest of the elements left. 
    The last element is set to 'padding_value'.
    """
    assert torch.all((idx >= 0) & (idx < arr.shape[1] - 1)), "Index out of bounds"
    assert idx.shape[0] == arr.shape[0], "Index and array must have the same number of rows"
    
    arr_copy = arr.clone()
    
    row_indices = torch.arange(arr.shape[0])

    # Fuse spans: [start of idx, end of idx+1]
    fused_spans = torch.stack([
        arr_copy[row_indices, idx, 0],
        arr_copy[row_indices, idx + 1, 1]
    ], dim=1)

    # Assign fused spans at idx positions
    arr_copy[row_indices, idx] = fused_spans

    # Shift elements left for each row
    for i in range(arr.shape[0]):
        fusion_index = idx[i].item()
        arr_copy[i, fusion_index+1:-1] = arr[i, fusion_index+2:]

    # Pad last element of each row
    arr_copy[:, -1] = padding_value

    return arr_copy, fused_spans

def overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    """
    Compute the overlap ratio between two spans.
    """
    start_a, end_a = a
    start_b, end_b = b
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    return overlap / (max(end_a, end_b) - min(start_a, start_b) + 1)

class Environment:
    def __init__(self, num_episodes, max_num_steps, success_weight:float, device: torch.device, symbol_freq: dict[str, float] = None, supervised: bool = False):
        self.symbol_freq = symbol_freq
        self.supervised = supervised
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
        self.rew : torch.Tensor = torch.zeros((num_episodes, max_num_steps), dtype=torch.float32, device=device)
        self.sym_rew : torch.Tensor = torch.zeros((num_episodes, max_num_steps), dtype=torch.float32, device=device)
        self.rhs_index : torch.Tensor = torch.full((num_episodes, max_num_steps), -1, dtype=torch.int32, device=device)

        self.step_count : int = 0
        self.done : torch.Tensor = torch.zeros(num_episodes, dtype=torch.bool, device=device)
        self.ep_len : torch.Tensor = torch.full((num_episodes,), max_num_steps, dtype=torch.int32, device=device)

    def reset(self, batch_sentences: torch.Tensor, batch_spans: list[dict[tuple[int, int], int]]):
        # batch_sentences: [num_episodes, sentence_length]
        self.state = torch.full(
            (self.num_episodes, self.max_num_steps + 1, batch_sentences.size(1)),
            -1,
            dtype=torch.int32,
            device=self.device
        )
        self.gt_spans = batch_spans
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
        self.sym_rew.zero_()
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
        # Print summary for first sentence
        #logger.info(f"GT spans: {self.gt_spans[0]}")
        #logger.info(f"Step {self.step_count}:")
        #logger.info(f"Current state: {self.state[0, self.step_count, :].tolist()}")
        #logger.info(f"Current spans: {self.spans_sentences[0, :].tolist()}")

        not_done = ~self.done
        not_done_idx = torch.arange(self.num_episodes, device=self.device)[not_done]
        #if not_done[0].item():
        #    logger.info(f"Pred positions: {positions[0].item()}, Pred symbols: {symbols[0].item()}")
        assert positions.shape == positions_log_probs.shape == symbols.shape == symbols_log_probs.shape == (not_done.sum().item(),), f"Shape mismatch: {positions.shape}, {positions_log_probs.shape}, {symbols.shape}, {symbols_log_probs.shape}, {(not_done.sum().item(),)}"

        current_state = self.state[not_done, self.step_count, :]
        current_spans = self.spans_sentences[not_done]

        self.state[:, self.step_count + 1, :] = self.state[:, self.step_count, :]
        if self.supervised:
            gt_positions, gt_symbols, gt_new_spans = self.gt_pos_sym_span()
            #if not_done[0].item():
            #    logger.info(f"GT positions: {gt_positions[0].item()}, GT symbols: {gt_symbols[0].item()}")

            next_sentence = fuse_and_shift(current_state, gt_positions, gt_symbols, -1)
            next_spans, new_spans = fuse_spans_and_shift(current_spans, gt_positions, -1)
            assert (gt_new_spans == new_spans).all(), f"New spans do not match gt_new_spans: {new_spans} != {gt_new_spans}"
        else:
            next_sentence = fuse_and_shift(current_state, positions, symbols, -1)
            next_spans, new_spans = fuse_spans_and_shift(current_spans, positions, -1)

        self.state[not_done, self.step_count + 1, :] = next_sentence
        self.spans_sentences[not_done, :, :] = next_spans

        for i in range(len(new_spans)):
            self.spans_lists[not_done_idx[i]].append((new_spans[i, 0].item(), new_spans[i, 1].item()))

        self.positions[not_done, self.step_count] = positions
        self.positions_log_probs[not_done, self.step_count] = positions_log_probs
        self.symbols[not_done, self.step_count] = symbols
        self.symbols_log_probs[not_done, self.step_count] = symbols_log_probs

        if self.supervised:
            row_indices = torch.arange(current_spans.shape[0])
            pred_new_spans = torch.stack([
                current_spans[row_indices, positions, 0],
                current_spans[row_indices, positions + 1, 1]
            ], dim=1)

            self.rew[not_done, self.step_count], self.sym_rew[not_done, self.step_count] = self.supervised_spans_reward_v2(not_done, pred_new_spans, gt_new_spans, symbols)
        else:
            self.rew[not_done, self.step_count], self.sym_rew[not_done, self.step_count] = self.supervised_spans_reward(not_done, new_spans, symbols)

        #logger.info(f"Position reward: {self.rew[0, self.step_count].item()}, Symbol reward: {self.sym_rew[0, self.step_count].item()}")

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
        batch_sym_rtgs = torch.zeros((self.num_episodes, self.ep_len.max()+1), dtype=torch.float32, device=self.device)

        discounted_reward = self.success_weight * self.success()
        for i in range(self.num_episodes):
            batch_rtgs[i, self.ep_len[i]] = discounted_reward[i]
            batch_sym_rtgs[i, self.ep_len[i]] = discounted_reward[i]
            for t in reversed(range(self.ep_len[i])):
                batch_rtgs[i, t] = self.rew[i, t] + gamma * batch_rtgs[i, t + 1]
                batch_sym_rtgs[i, t] = self.sym_rew[i, t] + gamma * batch_sym_rtgs[i, t + 1]
        return batch_rtgs, batch_sym_rtgs
    
    def supervised_spans_reward(self, not_done: torch.Tensor, new_spans: torch.Tensor, new_symbols: torch.Tensor):
        """
        Compute the reward for the new spans based on the ground truth spans.
        """
        gt_spans = [span for span, nd in zip(self.gt_spans, not_done) if nd]
        span_reward = torch.zeros(len(new_spans), dtype=torch.float32, device=self.device)
        sym_reward = torch.zeros(len(new_spans), dtype=torch.float32, device=self.device)

        for i in range(len(new_spans)):
            sym = gt_spans[i].get((new_spans[i,0].item(), new_spans[i,1].item()), None)
            span_reward[i] = sym is not None
            spans_with_pred_sym = [span for span, symm in gt_spans[i].items() if symm == new_symbols[i].item()]
            if len(spans_with_pred_sym) > 0:
                sym_reward[i] = max([overlap_ratio((new_spans[i,0].item(), new_spans[i,1].item()), span) for span in spans_with_pred_sym]) if self.symbol_freq is None else max([overlap_ratio((new_spans[i,0].item(), new_spans[i,1].item()), span) for span in spans_with_pred_sym])*(1.0-self.symbol_freq[IDX_TO_NT_TAG[new_symbols[i].item()]])
            else:
                sym_reward[i] = 0

        return span_reward, sym_reward
    
    def supervised_spans_reward_v2(self, not_done: torch.Tensor, pred_new_spans:torch.Tensor, gt_new_spans: torch.Tensor, new_symbols: torch.Tensor):
        """
        Compute the reward for the new spans based on the ground truth spans.
        """
        num_sentences = pred_new_spans.shape[0]
        gt_spans = [span for span, nd in zip(self.gt_spans, not_done) if nd]

        span_reward = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
        sym_reward = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)

        for i in range(num_sentences):
            sym_for_pred_span = gt_spans[i].get((pred_new_spans[i,0].item(), pred_new_spans[i,1].item()), None)
            sym_for_gt_span = gt_spans[i].get((gt_new_spans[i,0].item(), gt_new_spans[i,1].item()), None)
            if sym_for_pred_span is not None:
                span_reward[i] = 1.0 if self.symbol_freq is None else (1.0-self.symbol_freq[IDX_TO_NT_TAG[sym_for_pred_span]])
            else:
                span_reward[i] = 0.0
            if sym_for_gt_span is not None:
                sym_reward[i] = new_symbols[i]==sym_for_gt_span if self.symbol_freq is None else (new_symbols[i]==sym_for_gt_span)*(1.0-self.symbol_freq[IDX_TO_NT_TAG[sym_for_gt_span]])
            else:
                sym_reward[i] = 0.0

        return span_reward, sym_reward
    

    def gt_pos_sym_span(self):
        not_done = ~self.done
        current_spans = self.spans_sentences[not_done]
        index_nd = torch.arange(self.num_episodes, device=self.device)[not_done]
        gt_spans = [span for span, nd in zip(self.gt_spans, not_done) if nd]
        positions = torch.zeros(len(gt_spans), dtype=torch.int32, device=self.device)
        symbols = torch.zeros(len(gt_spans), dtype=torch.int32, device=self.device)
        new_spans = torch.zeros((len(gt_spans),2), dtype=torch.int32, device=self.device)
        for i in range(len(gt_spans)):
            found = False
            for idx, (span1, span2) in enumerate(zip(current_spans[i, :-1], current_spans[i, 1:])):
                span = (span1[0].item(), span2[1].item())
                sym = gt_spans[i].get((span[0], span[1]), None)
                if sym is not None:
                    found = True
                    positions[i] = idx
                    symbols[i] = sym
                    new_spans[i, 0] = span[0]
                    new_spans[i, 1] = span[1]
                    break
            if not found:
                raise ValueError(f"Span not found in gt_spans: {gt_spans[i]} for current span sentence {current_spans[i]} and state {self.state[index_nd[i], self.step_count, :]} and initial sentence {self.state[index_nd[i], 0, :]} at timestep {self.step_count}")
        return positions, symbols, new_spans
            
    
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
        batch_sym_rewards = []
        mask_position = []
        mask_symbol = []
        rews_to_go, sym_rews_to_go = self.compute_rtgs(gamma)
        for i in range(self.num_episodes):
            batch_states += self.state[i, :self.ep_len[i], :].tolist()
            batch_positions += self.positions[i, :self.ep_len[i]].tolist()
            batch_positions_log_probs += self.positions_log_probs[i, :self.ep_len[i]].tolist()
            batch_rewards += (rews_to_go[i, :self.ep_len[i]]).tolist()
            batch_symbols += self.symbols[i, :self.ep_len[i]].tolist()
            batch_symbols_log_probs += self.symbols_log_probs[i, :self.ep_len[i]].tolist()
            batch_sym_rewards += (sym_rews_to_go[i, :self.ep_len[i]]).tolist()
            #mask_position += [1] * self.ep_len[i] + [0]
            #mask_symbol += [0] + [1] * self.ep_len[i]
        return torch.tensor(batch_states, dtype=torch.int32, device=self.device), \
               torch.tensor(batch_rewards, dtype=torch.float32, device=self.device), \
                torch.tensor(batch_positions, dtype=torch.int32, device=self.device), \
                torch.tensor(batch_positions_log_probs, dtype=torch.float32, device=self.device), \
                torch.tensor(batch_symbols, dtype=torch.int32, device=self.device), \
                torch.tensor(batch_symbols_log_probs, dtype=torch.float32, device=self.device), \
                torch.tensor(batch_sym_rewards, dtype=torch.float32, device=self.device), \
                rews_to_go, \
                #torch.tensor(mask_position, dtype=torch.bool, device=self.device), \
                #torch.tensor(mask_symbol, dtype=torch.bool, device=self.device), \
    

    def rollout(self, actor_critic: ActorCritic, batch_sentences: torch.Tensor, batch_spans: list[dict[tuple[int, int]]], evaluate: bool = False):
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
            if self.supervised:
                gt_positions, _, _ = self.gt_pos_sym_span()
                position, position_log_prob, symbol, symbol_log_prob, _ = actor_critic.act(current_states, max_prob=evaluate, gt_positions=gt_positions)
            else:
                position, position_log_prob, symbol, symbol_log_prob, _ = actor_critic.act(current_states, max_prob=evaluate)
            assert position.shape == position_log_prob.shape == symbol.shape == symbol_log_prob.shape == (not_done.sum().item(),), f"Shape mismatch: {position.shape}, {position_log_prob.shape}, {symbol.shape}, {symbol_log_prob.shape}, {(not_done.sum().item(),)}"

            self.step(position, position_log_prob, symbol, symbol_log_prob)

            if self.stop():
                break
        
        logger.info(f"Rollout done. {self.num_episodes} episodes ran in {time.time() - time_s: .2f} secs")

