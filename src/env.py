import torch
import torch.nn.functional as F
from typing import Optional
from logging import getLogger
from actor_critic import ActorCritic
from grammar_env.corpus.sentence import Sentence
from n_gram import NGram
import time
from enum import Enum

logger = getLogger(__name__)

IDX_TO_NT_TAG = {
    0: 'X', 1: 'ADJP', 2: 'ADVP', 3: 'CONJP', 4: 'FRAG', 5: 'INTJ', 6: 'LST', 7: 'NAC', 8: 'NP',
    9: 'NX', 10: 'PP', 11: 'PRN', 12: 'PRT', 13: 'QP', 14: 'RRC', 15: 'S', 16: 'SBAR', 17: 'SBARQ',
    18: 'SINV', 19: 'SQ', 20: 'UCP', 21: 'VP', 22: 'WHADJP', 23: 'WHADVP', 24: 'WHNP', 25: 'WHPP'
}

class UpdateMode(Enum):
    ACTION = 0
    GT_POS = 1
    GT_POS_DEFAULT_SYM = 2
    GT_POS_GT_SYM = 3
    BEST_SYM = 4


def highlight_list_element(lst, index : int, additional_index: Optional[int] = None):
    if index < 0 or index >= len(lst):
        raise IndexError("Index out of range.")
    color='\033[31m'
    second_color='\033[32m'
    reset='\033[0m'
    # Build string representation
    parts = []
    for i, x in enumerate(lst):
        if i == index:
            parts.append(f"{color}{x}{reset}")  # highlight
        elif i == additional_index:
            parts.append(f"{second_color}{x}{reset}")  # highlight
        else:
            parts.append(str(x))
    print("[ " + ", ".join(parts) + " ]")

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
    def __init__(self, max_num_steps, success_weight:float, device: torch.device, symbol_freq: dict[str, float] = None, n_gram: NGram = None):
        self.symbol_freq = symbol_freq
        self.n_gram = n_gram
        self.device = device
        self.max_num_steps = max_num_steps
        self.success_weight = success_weight

        self.state : torch.Tensor = None # (batch_size, max_seq_len, embedding_dim)
        self.spans_sentences: torch.Tensor = None # (batch_size, max_seq_len, 2) list of spans associated with current constituents
        self.spans_lists: list[dict[tuple[int, int], int]] = None # (batch_size, DICT) list of dicts mapping spans to symbols for each sentence in the batch
        self.sentence_lengths: torch.Tensor = None # (batch_size,)

        self.action_positions : torch.Tensor = None # (batch_size, max_num_steps) with padding -1
        self.positions_log_probs : torch.Tensor = None # (batch_size, max_num_steps) with padding -inf
        self.action_symbols : torch.Tensor = None # (batch_size, max_num_steps) with padding -1
        self.symbols_log_probs : torch.Tensor = None # (batch_size, max_num_steps) with padding -inf
        self.update_positions : torch.Tensor = None # (batch_size, max_num_steps) with padding -1
        self.update_symbols : torch.Tensor = None # (batch_size, max_num_steps) with padding -1
        self.rew : torch.Tensor = None # (batch_size, max_num_steps) with padding 0
        self.sym_rew : torch.Tensor = None # (batch_size, max_num_steps) with padding 0
        self.state_val: torch.Tensor = None # (batch_size, max_num_steps) with padding 0.0

        self.step_count : int = 0
        self.done : torch.Tensor = None
        self.ep_len : torch.Tensor = None
        self.cls_tokens: Optional[torch.Tensor] = None

        self.update_mode: UpdateMode = UpdateMode.ACTION
        self.verbose: bool = False

    def reset(self, batch_sentences: list[Sentence], batch_s_embeddings: torch.Tensor):
        self.num_episodes = len(batch_sentences)
        self.sentences = batch_sentences
        self.state = batch_s_embeddings.to(self.device)
        self.cls_tokens = torch.zeros((self.num_episodes, batch_s_embeddings.shape[2]), dtype=torch.float32, device=self.device)
        self.gt_spans = [s.binary_gt_spans for s in batch_sentences]
        self.sentence_lengths = torch.tensor([len(sentence.symbols) for sentence in batch_sentences], device=self.device, dtype=torch.int32)
        max_seq_len = self.sentence_lengths.max().item()

        self.spans_sentences = torch.full(
            (self.num_episodes, max_seq_len, 2),
            -1,
            dtype=torch.int32,
            device=self.device
        )

        valid_symbols = torch.arange(max_seq_len, device=self.device, dtype=torch.int32)[None, :] < self.sentence_lengths[:, None]
        start_spans = torch.arange(max_seq_len, device=self.device, dtype=torch.int32)
        all_start_spans = torch.stack([start_spans, start_spans], dim=1)  # [length, 2]
        all_start_spans = all_start_spans.unsqueeze(0).repeat(self.num_episodes, 1, 1)
        self.spans_sentences[valid_symbols] = all_start_spans[valid_symbols]

        self.spans_lists : list[dict[tuple[int, int], int]] = [{} for _ in range(self.num_episodes)]

        self.action_positions = torch.full((self.num_episodes, self.max_num_steps), -1, dtype=torch.int32, device=self.device)
        self.positions_log_probs = torch.full((self.num_episodes, self.max_num_steps), float('-inf'), dtype=torch.float32, device=self.device)
        self.action_symbols = torch.full((self.num_episodes, self.max_num_steps), -1, dtype=torch.int32, device=self.device)
        self.symbols_log_probs = torch.full((self.num_episodes, self.max_num_steps), float('-inf'), dtype=torch.float32, device=self.device)
        self.update_positions = torch.full((self.num_episodes, self.max_num_steps), -1, dtype=torch.int32, device=self.device)
        self.update_symbols = torch.full((self.num_episodes, self.max_num_steps), -1, dtype=torch.int32, device=self.device)
        self.rew = torch.zeros((self.num_episodes, self.max_num_steps), dtype=torch.float32, device=self.device)
        self.sym_rew = torch.zeros((self.num_episodes, self.max_num_steps), dtype=torch.float32, device=self.device)
        self.state_val = torch.zeros((self.num_episodes, self.max_num_steps), dtype=torch.float32, device=self.device)
        self.step_count = 0
        self.done = torch.zeros(self.num_episodes, dtype=torch.bool, device=self.device)
        self.ep_len = torch.full((self.num_episodes,), self.max_num_steps, dtype=torch.int32, device=self.device)
        self.check_done()


    def stop(self):
        return self.step_count >= self.max_num_steps - 1 or self.done.all()

    def check_done(self):
        done = (self.spans_sentences[:, 1:, 0] == -1).all(dim=1)
        assert (~done & self.done).sum() == 0, f"Some episodes marked as done but do not qulify as such"
        self.done = done
        self.ep_len[done] = torch.minimum(self.ep_len[done], torch.tensor(self.step_count, device=self.device))

    def print_parse(self, current_spans: torch.Tensor, current_mask: torch.Tensor):
        not_done_idx = torch.arange(self.num_episodes, device=self.device)[~self.done]
        print_idx = not_done_idx[0] # Just print the first sentence in the batch

        initial_spans = current_spans[0]
        initial_mask = current_mask[0]

        pos_action = self.action_positions[print_idx, self.step_count].item()
        sym_action = self.action_symbols[print_idx, self.step_count].item()
        update_pos = self.update_positions[print_idx, self.step_count].item()
        update_sym = self.update_symbols[print_idx, self.step_count].item()
        pos_rew = self.rew[print_idx, self.step_count].item()
        sym_rew = self.sym_rew[print_idx, self.step_count].item()

        final_spans = self.spans_sentences[print_idx]

        print("\nInitial spans:\n")
        highlight_list_element(initial_spans[initial_mask].tolist(), pos_action, additional_index=update_pos)
        print(f"\nAction position: {pos_action}, Action symbol: {sym_action}, Update position: {update_pos}, Update symbol: {update_sym}\n")
        print(f"Position reward: {pos_rew}, Symbol reward: {sym_rew}\n")
        print(self.gt_spans[print_idx])
        print("Final spans:\n")
        highlight_list_element(final_spans[final_spans[:,0]!=-1].tolist(), pos_action)




    def step(self, action_positions: torch.Tensor, positions_log_probs: torch.Tensor, action_symbols: torch.Tensor, symbols_log_probs: torch.Tensor, update_positions: torch.Tensor, update_symbols: torch.Tensor, state_value: torch.Tensor, args: dict = None, evaluate: bool = False):
        new_constituents, drp_new_constituents = args["new_constituent"], args["drp_new_constituent"]

        not_done = ~self.done
        not_done_idx = torch.arange(self.num_episodes, device=self.device)[not_done]
        not_done_range = torch.arange(not_done.sum().item(), device=self.device)
        assert action_positions.shape == positions_log_probs.shape == action_symbols.shape == symbols_log_probs.shape == (not_done.sum().item(),), f"Shape mismatch: {action_positions.shape}, {positions_log_probs.shape}, {action_symbols.shape}, {symbols_log_probs.shape}, {(not_done.sum().item(),)}"

        current_state = self.state[not_done]
        current_spans = self.spans_sentences[not_done]
        current_mask = self.get_mask()[not_done]

        next_sentence = fuse_and_shift(current_state, update_positions, new_constituents, 0.)
        next_spans, new_spans = fuse_spans_and_shift(current_spans, update_positions, -1)

        self.state[not_done] = next_sentence
        self.spans_sentences[not_done] = next_spans

        for i in range(len(new_spans)):
            self.spans_lists[not_done_idx[i]][(new_spans[i, 0].item(), new_spans[i, 1].item())] = action_symbols[i].item()

        self.action_positions[not_done, self.step_count] = action_positions
        self.positions_log_probs[not_done, self.step_count] = positions_log_probs
        self.action_symbols[not_done, self.step_count] = action_symbols
        self.symbols_log_probs[not_done, self.step_count] = symbols_log_probs
        self.update_positions[not_done, self.step_count] = update_positions
        self.update_symbols[not_done, self.step_count] = update_symbols
        self.state_val[not_done, self.step_count] = state_value

        if not evaluate:
            
            action_spans = torch.stack((current_spans[not_done_range, action_positions, 0], current_spans[not_done_range, action_positions + 1, 1]), dim=1)
            self.rew[not_done, self.step_count] = self.supervised_spans_reward(not_done, action_spans)
            #self.rew[not_done, self.step_count] = self.contrastive_loss_vec(current_state, action_positions, current_mask)
            #self.rew[not_done, self.step_count] = self.overlap_sym_reward(not_done, new_spans, action_symbols)
            #self.rew[not_done, self.step_count] = self.n_gram_reward(not_done, current_spans[not_done_range, action_positions], current_spans[not_done_range, action_positions + 1])
            #self.rew[not_done, self.step_count] = self.n_gram_attraction(not_done, action_positions, current_spans)
            
            #self.sym_rew[not_done, self.step_count] = self.supervised_sym_reward(not_done, new_spans, action_symbols)
            self.sym_rew[not_done, self.step_count] = self.overlap_sym_reward(not_done, new_spans, action_symbols)
            #self.sym_rew[not_done, self.step_count] = (F.cosine_similarity(drp_new_constituents, new_constituents, dim=1)+1)/2

            #self.rew[not_done, self.step_count] = 0.5 * (self.rew[not_done, self.step_count] + self.sym_rew[not_done, self.step_count])
            '''
            if self.step_count > 0:
                cls_cossim = self.reward_cossim_cls(not_done, cls_token)
                #self.rew[not_done, self.step_count-1] = cls_cossim
                self.sym_rew[not_done, self.step_count-1] = cls_cossim
            '''
            #self.cls_tokens[not_done] = cls_token

        if self.verbose:
            self.print_parse(current_spans, current_mask)
        self.step_count += 1
        self.check_done()

    def get_state(self):
        not_done = ~self.done
        current_state = self.state[not_done]
        return current_state, not_done
    
    def get_mask(self) -> torch.Tensor:
        """
        Get the mask for the valid symbols/padding in each sentence.
        """
        return self.spans_sentences[:, :, 0] != -1

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
    
    def supervised_spans_reward(self, not_done: torch.Tensor, new_spans: torch.Tensor):
        """
        Compute the reward for the new spans based on the ground truth spans.
        """
        gt_spans = [span for span, nd in zip(self.gt_spans, not_done) if nd]
        span_reward = torch.zeros(len(new_spans), dtype=torch.float32, device=self.device)

        for i in range(len(new_spans)):
            sym = gt_spans[i].get((new_spans[i,0].item(), new_spans[i,1].item()), None)
            span_reward[i] = sym is not None

        return span_reward

    def overlap_sym_reward(self, not_done: torch.Tensor, new_spans: torch.Tensor, new_symbols: torch.Tensor):
        """
        Compute the reward for the chosen symbol assuming symbol actor was given "new_spans" as input.
        Reward is given by highest overlap ratio with any GT span with the same symbol.
        """
        gt_spans = [span for span, nd in zip(self.gt_spans, not_done) if nd]
        sym_reward = torch.zeros(len(new_spans), dtype=torch.float32, device=self.device)

        for i in range(len(new_spans)):
            spans_with_pred_sym = [span for span, symm in gt_spans[i].items() if symm == new_symbols[i].item()]
            if len(spans_with_pred_sym) > 0:
                nt_factor = 1.0 if self.symbol_freq is None else min(10.0, 1/(len(self.symbol_freq.keys())*self.symbol_freq.get(new_symbols[i].item(), 0.0) + 1e-9))
                sym_reward[i] = max([overlap_ratio((new_spans[i,0].item(), new_spans[i,1].item()), span) for span in spans_with_pred_sym]) * nt_factor
            else:
                sym_reward[i] = 0

        return sym_reward
    
    def supervised_sym_reward(self, not_done: torch.Tensor, new_spans: torch.Tensor, new_symbols: torch.Tensor):
        """
        Compute the reward for the chosen symbols assuming symbol actor was given "new_spans" as input and that new_spans is in the GT spans.
        """
        gt_spans = [span for span, nd in zip(self.gt_spans, not_done) if nd]
        sym_reward = torch.zeros(len(new_spans), dtype=torch.float32, device=self.device)

        for i in range(len(new_spans)):
            sym = gt_spans[i].get((new_spans[i,0].item(), new_spans[i,1].item()), None)
            nt_factor = 1.0 if self.symbol_freq is None else min(10.0, 1/(len(self.symbol_freq.keys())*self.symbol_freq.get(new_symbols[i].item(), 0.0) + 1e-9))
            sym_reward[i] = float(sym == new_symbols[i].item())* nt_factor

        return sym_reward

    def supervised_update_sym(self, not_done: torch.Tensor, new_spans: torch.Tensor):
        """
        Determines the most appropriate symbol for given span based in the "closest" ground truth span (shortest covering span).
        """
        gt_spans = [span for span, nd in zip(self.gt_spans, not_done) if nd]
        symbols = torch.zeros(len(new_spans), dtype=torch.int32, device=self.device)

        for i in range(len(new_spans)):
            new_span = (new_spans[i,0].item(), new_spans[i,1].item())
            shortest_covering_span = min([span for span in gt_spans[i].keys() if span[0] <= new_span[0] and span[1] >= new_span[1]], key=lambda x: x[1]-x[0], default=None)
            assert shortest_covering_span is not None, f"Shortest covering span not found for {new_span} in {gt_spans[i]}"
            symbols[i] = gt_spans[i][shortest_covering_span]

        return symbols
    
    def reward_cossim_cls(self, not_done: torch.Tensor, cls_token: torch.Tensor):
        """
        Compute the reward for the new constituents based on the similarity between CLS tokens before and after update.
        """
        return (F.cosine_similarity(cls_token, self.cls_tokens[not_done], dim=1)+1)/2
    
    def contrastive_loss(self, state: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor):
        margin = 0.5
        losses = torch.zeros(state.shape[0], dtype=torch.float32, device=self.device)
        for i, (sen, pos, m) in enumerate(zip(state, positions, mask)):
            valid_sen = sen[m]
            pair_dist = torch.cdist(valid_sen, valid_sen, p=2)  # Compute pairwise distances
            labels = torch.zeros_like(pair_dist, dtype=torch.bool, device=self.device)
            labels.fill_diagonal_(True)  # Set diagonal to True (same token pairs)
            labels[pos, pos+1] = True  # Set the position pair to True (fused token pair)
            labels[pos+1, pos] = True  # Set the position pair to True (fused token pair)
            similar_pair_loss = (pair_dist[labels]**2).mean()
            dissimilar_pair_loss = (torch.clamp(margin - pair_dist[~labels], min=0)**2).mean() if torch.any(~labels) else 0.0
            loss = 0.5* (similar_pair_loss + dissimilar_pair_loss)
            losses[i] = loss
        assert not torch.isnan(losses).any(), f"NaN found in losses: {losses}"
        return torch.exp(-losses)
    
    def contrastive_loss_vec(self, state: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor):
        """
        Compute the contrastive loss for a batch of states.
        """
        margin = 0.5
        batch_size, seq_len, dim = state.shape

        # Mask invalid positions
        all_valid_pairs = mask[:, None, :] & mask[:, :, None]  # (B, N, N)
        
        # Compute pairwise distances per batch (B, N, N)
        pair_dists = torch.cdist(state, state, p=2)
        
        # Build label masks (B, N, N)
        diag_mask = torch.eye(seq_len, device=state.device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
        pos_mask = torch.zeros_like(diag_mask)
        pos_mask[torch.arange(batch_size), positions, positions+1] = True
        pos_mask[torch.arange(batch_size), positions+1, positions] = True

        labels = diag_mask | pos_mask

        # Similar pairs
        valid_and_similar = labels & all_valid_pairs
        sq_dists = (pair_dists**2).masked_fill(~valid_and_similar, 0.0)  # Set dissimilar pairs or invalid pairs to zero distance
        count = valid_and_similar.sum(dim=(1, 2))  # Count similar pairs per batch
        similar_loss = sq_dists.sum(dim=(1, 2)) / count.clamp(min=1)  # Avoid division by zero

        # Dissimilar pairs
        valid_and_dissimilar = ~labels & all_valid_pairs
        sq_dists = (torch.clamp(margin - pair_dists, min=0)**2).masked_fill(~valid_and_dissimilar, 0.0)  # Set similar pairs or invalid pairs to zero distance
        count = (valid_and_dissimilar).sum(dim=(1, 2))  # Count dissimilar pairs per batch
        dissimilar_loss = sq_dists.sum(dim=(1, 2)) / count.clamp(min=1)  # Avoid division by zero

        total_loss = 0.5 * (similar_loss + dissimilar_loss)

        return torch.exp(-total_loss)

    def n_gram_reward(self, not_done: torch.Tensor, left_spans: torch.Tensor, right_spans: torch.Tensor):
        """
        Compute the n-gram reward for the new spans based on the n-gram model.
        """
        num_sentences = left_spans.shape[0]
        rewards = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
        if self.n_gram is None:
            return rewards
        sentences = [s for s, nd in zip(self.sentences, not_done) if nd]

        def left_right_prob(idx : int):
            left_ngram = sentences[idx].pos_tags[left_spans[idx, 0].item():left_spans[idx, 1].item() + 1]
            right_ngram = sentences[idx].pos_tags[right_spans[idx, 0].item():right_spans[idx, 1].item() + 1]
            return self.n_gram.compute_prob(tuple(left_ngram), tuple(right_ngram))
        for i in range(num_sentences):
            rewards[i] = left_right_prob(i)
        return rewards
    
    def n_gram_attraction(self, not_done: torch.Tensor, positions: torch.Tensor, spans_sen: torch.Tensor):
        if self.n_gram is None:
            return torch.zeros(positions.shape[0], dtype=torch.float32, device=self.device)
        indices = torch.arange(positions.shape[0], device=self.device)
        current_sen_lengths = torch.sum(spans_sen[:, :, 0] > -1, dim=1)
        sentences = [s for i,s in enumerate(self.sentences) if not_done[i]]
        left_spans = spans_sen[indices, positions]
        center_spans = spans_sen[indices, positions + 1]

        valid_right =  positions + 2 < current_sen_lengths
        right_spans = torch.zeros_like(left_spans)
        right_spans[valid_right] = spans_sen[indices[valid_right], positions[valid_right] + 2]

        attraction = torch.zeros(positions.shape[0], dtype=torch.float32, device=self.device)
        for  i in range(positions.shape[0]):
            left_constituents = tuple(sentences[i].pos_tags[left_spans[i, 0].item():left_spans[i, 1].item() + 1])
            center_constituents = tuple(sentences[i].pos_tags[center_spans[i, 0].item():center_spans[i, 1].item() + 1])
            if valid_right[i]:
                right_constituents = tuple(sentences[i].pos_tags[right_spans[i, 0].item():right_spans[i, 1].item() + 1])
            else:
                right_constituents = tuple()
            attraction[i] = self.n_gram.compute_attraction(left_constituents, center_constituents, right_constituents)
        return attraction
            
            

    
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
        batch_state_val = []
        batch_positions = []
        batch_positions_log_probs = []
        batch_rewards = []
        batch_symbols = []
        batch_symbols_log_probs = []
        batch_sym_rewards = []
        mask_before = []
        mask_after = []
        rews_to_go, sym_rews_to_go = self.compute_rtgs(gamma)
        for i in range(self.num_episodes):
            batch_state_val += self.state_val[i, :self.ep_len[i]+1].tolist()
            batch_positions += self.action_positions[i, :self.ep_len[i]].tolist()
            batch_positions_log_probs += self.positions_log_probs[i, :self.ep_len[i]].tolist()
            batch_rewards += (rews_to_go[i, :self.ep_len[i]]).tolist()
            batch_symbols += self.action_symbols[i, :self.ep_len[i]].tolist()
            batch_symbols_log_probs += self.symbols_log_probs[i, :self.ep_len[i]].tolist()
            batch_sym_rewards += (sym_rews_to_go[i, 0:self.ep_len[i]+0]).tolist()
            mask_before += [1] * self.ep_len[i] + [0]
            mask_after += [0] + [1] * self.ep_len[i]
        return torch.tensor(batch_state_val, dtype=torch.float32, device=self.device), \
                torch.tensor(batch_positions, dtype=torch.int32, device=self.device), \
                torch.tensor(batch_positions_log_probs, dtype=torch.float32, device=self.device), \
                torch.tensor(batch_rewards, dtype=torch.float32, device=self.device), \
                torch.tensor(batch_symbols, dtype=torch.int32, device=self.device), \
                torch.tensor(batch_symbols_log_probs, dtype=torch.float32, device=self.device), \
                torch.tensor(batch_sym_rewards, dtype=torch.float32, device=self.device), \
                rews_to_go, \
                torch.tensor(mask_before, dtype=torch.bool, device=self.device), \
                torch.tensor(mask_after, dtype=torch.bool, device=self.device), \
    

    def get_update_pos_sym(self, position_action: torch.Tensor, symbol_action: torch.Tensor, gt_position: torch.Tensor, gt_symbol: torch.Tensor):
        if self.update_mode == UpdateMode.ACTION:
            return position_action, symbol_action
        elif self.update_mode == UpdateMode.GT_POS:
            return gt_position, symbol_action
        elif self.update_mode == UpdateMode.GT_POS_DEFAULT_SYM:
            return gt_position, torch.zeros_like(symbol_action, dtype=torch.int32, device=self.device)
        elif self.update_mode == UpdateMode.GT_POS_GT_SYM:
            return gt_position, gt_symbol
        elif self.update_mode == UpdateMode.BEST_SYM:
            _, not_done = self.get_state()
            current_spans_sentences = self.spans_sentences[not_done]
            not_done_range = torch.arange(not_done.sum().item(), device=self.device)
            action_spans = torch.stack([current_spans_sentences[not_done_range, position_action, 0], current_spans_sentences[not_done_range, position_action + 1, 1]], dim=1)
            update_symbol = self.supervised_update_sym(not_done, action_spans)
            return position_action, update_symbol
        raise ValueError(f"Unknown update mode: {self.update_mode}")



    def rollout(self, actor_critic: ActorCritic, batch_sentences: list[Sentence], evaluate: bool = False):
        """
        Collect batch of data from simulation.
        As PPO is an on-policy algorithm, we need to collect a fresh batch
        of data each time we iterate the actor/critic networks.

        :param actor_critic: ActorCritic model to use for the rollout.
        :param batch_sentences: Batch of sentences to use for the rollout (list[Sentence]).
        :param batch_spans: List of spans for each sentence in the batch.
        :param evaluate: If True, the model will not sample actions but take the most probable ones.
        """
        time_s = time.time()

        batch_s_embeddings = actor_critic.encode_sentence(batch_sentences)

        # Reset the environment.
        self.reset(batch_sentences, batch_s_embeddings)

        cls_tokens = actor_critic.get_cls_token(self.state, self.get_mask())
        self.cls_tokens = cls_tokens

        # Run an episode for max_timesteps_per_episode timesteps
        ep_t: int = 0
        for ep_t in range(self.max_num_steps):
            if self.update_mode in {UpdateMode.GT_POS, UpdateMode.GT_POS_DEFAULT_SYM, UpdateMode.GT_POS_GT_SYM}:
                gt_pos, gt_sym, gt_new_spans = self.gt_pos_sym_span() # Supervised span and symbol
            else:
                gt_pos, gt_sym, gt_new_spans = None, None, None
            current_states, not_done = self.get_state()
            not_done_range = torch.arange(not_done.sum().item(), device=self.device)
            position, position_log_prob, symbol, symbol_log_prob, state_value = actor_critic.act(current_states, self.get_mask()[not_done], position_update=gt_pos)
            #cls_token = actor_critic.get_cls_token(current_states, self.get_mask()[not_done])
            assert position.shape == position_log_prob.shape == symbol.shape == symbol_log_prob.shape == state_value.shape == (not_done.sum().item(),), f"Shape mismatch: {position.shape}, {position_log_prob.shape}, {symbol.shape}, {symbol_log_prob.shape}, {state_value.shape}, {(not_done.sum().item(),)}"

            update_position, update_symbol = self.get_update_pos_sym(position, symbol, gt_pos, gt_sym)

            # Compute new constituents embeddings
            left = current_states[not_done_range, update_position, :]
            right = current_states[not_done_range, update_position + 1, :]
            emb_sym = actor_critic.tag_embedder(update_symbol)
            new_constituent = actor_critic.fuse_constituents(emb_sym, left, right)
            # Constituent embedding with dropout for additional objective
            drp_new_constituent = actor_critic.fuse_constituents(emb_sym, left, right, dropout=True)
            #new_constituent = new_constituent if evaluate else drp_new_constituent

            args = {"new_constituent": new_constituent, "drp_new_constituent": drp_new_constituent}
            self.step(position, position_log_prob, symbol, symbol_log_prob, update_position, update_symbol, state_value, args=args, evaluate=evaluate)


            if self.stop():
                break
        

        #last_cls_token = actor_critic.get_cls_token(self.state, self.get_mask())
        #self.rew[torch.arange(self.num_episodes, device=self.device), self.ep_len-1] = self.reward_cossim_cls(torch.ones((self.num_episodes,), dtype=torch.bool, device=self.device), last_cls_token)
        #self.sym_rew[torch.arange(self.num_episodes, device=self.device), self.ep_len-1] = self.reward_cossim_cls(torch.ones((self.num_episodes,), dtype=torch.bool, device=self.device), last_cls_token)
        
        logger.info(f"Rollout done. {self.num_episodes} episodes ran in {time.time() - time_s: .2f} secs")

    def replay(self, actor_critic: ActorCritic, b_add_obj:bool = False):
        """
        Replay the preceding simulation, in order to collect current log probabilities and state values for PPO loss.

        :param actor_critic: ActorCritic model to use for the replay.
        """
        actor_critic.train()
        batch_s_embeddings = actor_critic.encode_sentence(self.sentences)
        self.state = batch_s_embeddings.to(self.device)
        max_seq_len = self.sentence_lengths.max().item()
        mask = torch.arange(max_seq_len, device=self.device)[None, :] < self.sentence_lengths[:, None]

        curr_pos_log_probs = torch.zeros((self.num_episodes, max_seq_len-1), dtype=torch.float32, device=self.device)
        curr_sym_log_probs = torch.zeros((self.num_episodes, max_seq_len-1), dtype=torch.float32, device=self.device)
        curr_pos_entropies = torch.zeros((self.num_episodes, max_seq_len-1), dtype=torch.float32, device=self.device)
        curr_sym_entropies = torch.zeros((self.num_episodes, max_seq_len-1), dtype=torch.float32, device=self.device)
        curr_values = torch.zeros((self.num_episodes, max_seq_len-1), dtype=torch.float32, device=self.device)
        

        # Use python list and then torch.stack for gradient tracking

        """
        curr_pos_log_probs = [[] for _ in range(self.num_episodes)]
        curr_sym_log_probs = [[] for _ in range(self.num_episodes)]
        curr_pos_entropies = [[] for _ in range(self.num_episodes)]
        curr_sym_entropies = [[] for _ in range(self.num_episodes)]
        curr_values = [[] for _ in range(self.num_episodes)]
        """

        add_obj = torch.tensor([0.0], device=self.device) # Additional objective for the PPO loss, e.g. contrastive loss
        nb_terms = 0
        # Run an episode for max_timesteps_per_episode timesteps
        ep_t: int = 0
        for ep_t in range(self.ep_len.max()):

            not_done = torch.full((self.num_episodes,), ep_t, dtype=torch.int32, device=self.device) < self.ep_len
            not_done_range = torch.arange(self.num_episodes, device=self.device)[not_done]

            current_states = self.state[not_done]
            current_act_pos = self.action_positions[not_done, ep_t]
            current_act_sym = self.action_symbols[not_done, ep_t]
            current_upd_pos = self.update_positions[not_done, ep_t]
            current_upd_sym = self.update_symbols[not_done, ep_t]
            pos_log_prob, pos_ent, sym_log_prob, sym_ent, state_value = actor_critic.evaluate(current_states, mask[not_done], current_act_pos, current_upd_pos, current_act_sym)

            curr_pos_log_probs[not_done, ep_t] , curr_sym_log_probs[not_done, ep_t], curr_pos_entropies[not_done, ep_t], curr_sym_entropies[not_done, ep_t], curr_values[not_done, ep_t] = pos_log_prob, sym_log_prob, pos_ent, sym_ent, state_value

            """
            for i in range(not_done.sum().item()):
                curr_pos_log_probs[not_done_range[i]].append(pos_log_prob[i])
                curr_sym_log_probs[not_done_range[i]].append(sym_log_prob[i])
                curr_pos_entropies[not_done_range[i]].append(pos_ent[i])
                curr_sym_entropies[not_done_range[i]].append(sym_ent[i])
                curr_values[not_done_range[i]].append(state_value[i])
            """

            # Update state with the new constituents
            left = current_states[torch.arange(not_done.sum().item()), current_upd_pos, :]
            right = current_states[torch.arange(not_done.sum().item()), current_upd_pos + 1, :]
            emb_sym = actor_critic.tag_embedder(current_upd_sym)
            new_constituent = actor_critic.fuse_constituents(emb_sym, left, right)

            if b_add_obj:
                drp_new_constituent = actor_critic.fuse_constituents(emb_sym, left, right, dropout=True)
                add_obj += self.contrastive_loss(current_states, current_act_pos, mask[not_done]).sum()
                nb_terms += not_done.sum().item()
                add_obj += ((F.cosine_similarity(drp_new_constituent, new_constituent, dim=1)+1)/2).sum()
                nb_terms += not_done.sum().item()
            self.state[not_done] = fuse_and_shift(current_states, current_upd_pos, new_constituent, 0.)

            lens = mask.sum(dim=1)
            mask[torch.arange(self.num_episodes, device=self.device), lens-1] = False

        # Prepare the data for the PPO loss
        add_obj = add_obj / (nb_terms+1)
        mask = torch.arange(max_seq_len-1, device=self.device)[None, :] < self.ep_len[:, None]
        """
        flat_curr_pos_log_probs = torch.cat([torch.stack(row) for row in curr_pos_log_probs if len(row) > 0], dim=0)
        flat_curr_sym_log_probs = torch.cat([torch.stack(row) for row in curr_sym_log_probs if len(row) > 0], dim=0)
        flat_curr_pos_entropies = torch.cat([torch.stack(row) for row in curr_pos_entropies if len(row) > 0], dim=0)
        flat_curr_sym_entropies = torch.cat([torch.stack(row) for row in curr_sym_entropies if len(row) > 0], dim=0)
        flat_curr_values = torch.cat([torch.stack(row) for row in curr_values if len(row) > 0], dim=0)
        """
        return curr_pos_log_probs[mask], curr_pos_entropies[mask], curr_sym_log_probs[mask], curr_sym_entropies[mask], curr_values[mask], add_obj
        # return flat_curr_pos_log_probs, flat_curr_pos_entropies, flat_curr_sym_log_probs, flat_curr_sym_entropies, flat_curr_values, add_obj
    

