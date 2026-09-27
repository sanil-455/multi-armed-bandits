import numpy as np

class UCB1Agent:
    def __init__(self, k):
        self.k = k # arm counts
        self.t = 0 # total no.of arm pulls made so far(global count)
        self.N = np.zeros(k, dtype=int) # arm pulls of particular arm unlike t which is global count
        self.Q = np.zeros(k) # current sample avg. reward estimate per arm

    def select_action(self):
	# check if any arm not tried
        untried = np.flatnonzero(self.N == 0)
	# if found return immediately as in paper its given so as to avoid division by zero
        if len(untried) > 0:
            return untried[0]
	
	# this line runs only when every arm has been pulled atleast once
	# add the exploitation term to the exploiyyaion

        confidence_bonus = np.sqrt(2 * np.log(self.t) / self.N)
        ucb_index = self.Q + confidence_bonus
        max_index = np.max(ucb_index)
        candidates = np.flatnonzero(ucb_index == max_index)
        return np.random.choice(candidates)

    def update(self, action, reward):
        self.t += 1
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])
