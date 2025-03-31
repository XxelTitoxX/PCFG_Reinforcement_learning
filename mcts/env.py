import numpy as np
from grammar import CompleteBinGrammar

'''
A state is represented as a sequence of rules, where each rule is represented by its index in the grammar table.
It corresponds to the partial "sentence-to-start-symbol" parsing of the current sentence.
An action is represented as a rule index to add to the current state, and corresponds to parsing the first symbol or pair of symbols that are recognized as the RHS of the rule.
'''

class Environment:
    def __init__(self, num_episodes, num_rules, state_size):
        self.num_rules = num_rules
        self.state_size = state_size
        self.state = np.zeros((num_episodes, state_size), dtype=np.float32) # Partially parsed sentence or rule sequence ?
        self.score = np.zeros((num_episodes, state_size), dtype=np.float32)
        self.step_count = 0
        self.done = np.zeros(num_episodes, dtype=bool)
        self.reset()

    def reset(self):
        self.state.fill(0)
        self.score.fill(0)
        self.step_count = 0
        self.done.fill(False)

    def check_done(self):
        pass


    def step(self, actions):
        self.check_done()
        not_done = ~self.done
        self.state[not_done, self.step_count] = actions


        
