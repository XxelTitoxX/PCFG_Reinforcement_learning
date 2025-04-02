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
    def __init__(self, num_episodes, num_rules, sentence_max_len, max_num_steps, bin_grammar: CompleteBinGrammar):
        self.bin_grammar = bin_grammar
        self.num_rules = num_rules
        self.state_size = sentence_max_len
        self.max_num_steps = max_num_steps
        self.state = np.zeros((num_episodes, self.max_num_steps+1, self.state_size), dtype=np.float32) # Partially parsed sentence
        self.actions = np.zeros((num_episodes, self.max_num_steps), dtype=np.int32) # Actions taken, rule indexes
        self.rew = np.zeros((num_episodes, self.max_num_steps), dtype=np.bool) # Actions that should be rewarded
        self.step_count = 0
        self.done = np.zeros(num_episodes, dtype=bool)
        self.reset()

    def reset(self):
        self.state.fill(-1)
        self.actions.fill(-1)
        self.rew.fill(0)
        self.step_count = 0
        self.done.fill(False)

    def init_state(self, sentences):
        self.state[:, 0, :] = sentences

    def stop(self):
        return self.step_count >= self.max_num_steps-1

    def check_done(self):
        '''
        Check if the only uninitialized symbol is start symbol = sentence fully parsed
        '''
        if self.step_count >= self.max_num_steps:
            self.done = np.ones(self.state.shape[0], dtype=bool)
            return
        self.done = self.state[:, self.step_count, 1:] == -1 & self.state[:, self.step_count, 0] == 0


    def step(self, actions):

        self.check_done()
        not_done = ~self.done
        # Apply actions to the states
        new_sentence, valid_parsing = self.bin_grammar.parsing_step(self.state[not_done, self.step_count, :], actions[not_done])
        self.state[:, self.step_count+1, :] = self.state[:, self.step_count, :]
        self.state[not_done, self.step_count+1, :] = new_sentence
        self.actions[:, self.step_count] = actions
        self.rew[:, self.step_count] = valid_parsing
        self.step_count += 1




        
