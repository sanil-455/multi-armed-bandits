import numpy as np

class ThompsonSamplingAgent:
    def __init__(self, k, rng=None):
        self.k = k
	# filling with 1s as in theory,start every arm with alpha=1 for uniform {Beta}(1,1) prior
        self.alpha = np.ones(k)
	# same for beta
        self.beta = np.ones(k)
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self):
	# sample from each arm's belief distribution
	# self.rng.beta(...) is numpy's built-in function for drawing a random sample from a Beta 
	# distribution given its alpha and beta params
        samples = self.rng.beta(self.alpha, self.beta)
	# returns highest sampled value this particular run(round)
        return int(np.argmax(samples))

    def update(self, action, reward):
	# if reward is 1 add 1 to alpha and if reward is 0 add 1 to beta
        self.alpha[action] += reward
        self.beta[action] += (1 - reward)
