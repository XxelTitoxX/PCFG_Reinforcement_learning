import numpy as np
from grammar import CompleteBinGrammar

'''
A state is represented as a partialy parsed sentence (in the sentence-to-start-symbol parsing order), that is to say a symbol sequense where each symbol is represented by its index in the grammar table.
An action is represented as a rule index, and corresponds to parsing the first symbol or pair of symbols that are recognized as the RHS of the rule.
'''

'''
CONVENTIONS :
- Symbol indexed by 0 is the start symbol
- Any uninitialized symbol or rule is represented by -1
'''

'''
In the context of supervised dependency parsing, the sentence is actually a sequence of pre-terminals.
'''

class Environment:
    def __init__(self, num_episodes, max_num_steps, bin_grammar: CompleteBinGrammar):
        self.bin_grammar = bin_grammar
        self.max_num_steps = max_num_steps
        self.num_episodes = num_episodes
        self.state = None # Partially parsed sentence
        self.actions = np.zeros((num_episodes, self.max_num_steps), dtype=np.int32) # Actions taken, rule indexes
        self.actions_log_probs = np.zeros((num_episodes, self.max_num_steps), dtype=np.float32) # Log probs of actions taken
        self.rew = np.zeros((num_episodes, self.max_num_steps), dtype=np.bool) # Actions that should be rewarded
        self.step_count = 0
        self.done = np.zeros(num_episodes, dtype=bool)
        self.ep_len = np.zeros(num_episodes, dtype=np.int32) # Length of each episode
        self.reset()

    def reset(self, sentence_max_len : int):
        self.state = np.zeros((self.num_episodes, self.max_num_steps+1, sentence_max_len), dtype=np.float32)
        self.state.fill(-1)
        self.actions.fill(-1)
        self.rew.fill(0)
        self.step_count = 0
        self.done.fill(False)
        self.ep_len.fill(self.max_num_steps)
        self.actions_log_probs.fill(0)

    def init_state(self, sentences):
        self.state[:, 0, :] = sentences

    def stop(self):
        return self.step_count >= self.max_num_steps-1 or self.done.all()

    def check_done(self):
        '''
        Check if the only uninitialized symbol is start symbol = sentence fully parsed
        '''
        if self.step_count >= self.max_num_steps:
            self.done = np.ones(self.state.shape[0], dtype=bool)
            return
        self.done = self.state[:, self.step_count, 1:] == -1 & self.state[:, self.step_count, 0] == 0
        self.ep_len[self.done] = np.minimum(self.ep_len[self.done], self.step_count)


    def step(self, actions, actions_log_probs):

        self.check_done()
        not_done = ~self.done
        # Apply actions to the states
        new_sentence, valid_parsing = self.bin_grammar.parsing_step(self.state[not_done, self.step_count, :], actions[not_done])
        self.state[:, self.step_count+1, :] = self.state[:, self.step_count, :]
        self.state[not_done, self.step_count+1, :] = new_sentence
        self.actions[:, self.step_count] = actions
        self.actions_log_probs[:, self.step_count] = actions_log_probs
        self.rew[:, self.step_count] = valid_parsing
        self.step_count += 1

    def get_state(self):
        self.check_done()
        not_done = ~self.done
        return self.state[:, self.step_count, :].copy(), not_done
    
    def get_rewards(self):
        return self.rew[:, :self.step_count].copy()
    
    def compute_rtgs(self) -> np.ndarray:
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.

        :param batch_rews: batch of rewards of an episode.
                            [[r_0, r_1, ..., r_T], [r'_0, r'_1, ..., r'_T], ..., [r''_0, r''_1, ..., r''_T]]
        :return:
            np.ndarray of shape (number of timesteps in batch)
            The Rewards-To-Go for each timestep in the batch.
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode).
        batch_rtgs: list[float] = []

        # Iterate through each episode
        for ep_rews in reversed(self.rews):
            discounted_reward: float = 0.  # The discounted reward so far

            # Iterate through all rewards in the episode.
            # We go backwards for smoother calculation of each discounted return
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.config.gamma
                batch_rtgs.insert(0, discounted_reward)

        return np.array(batch_rtgs, dtype=np.float32)
    






        
