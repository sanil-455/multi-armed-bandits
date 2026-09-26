import numpy as np
class OptimisticGreedyAgent:
    def __init__(self, k, initial_value=5.0, epsilon=0.0, step_size=0.1, rng=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = step_size
        self.rng = rng if rng is not None else np.random.default_rng()
	# builds an array of length k with every entry x
        self.Q = np.full(k, float(initial_value)) 
	# an array of k zeros that track how many times each arm has been pulled
        self.N = np.zeros(k, dtype=int)

    def select_action(self):
	# this is anyway false by default as epsilon value is set to zero deafult
	# just for future experimentation
        if self.rng.random() < self.epsilon:
	# select any random branch if true
            return self.rng.integers(self.k)
        max_q = np.max(self.Q)
        candidates = np.flatnonzero(self.Q == max_q)
        return self.rng.choice(candidates)

    def update(self, action, reward):
	# an array pulling counts per arm
        self.N[action] += 1
        self.Q[action] += self.alpha * (reward - self.Q[action])
