import numpy as np

class EXP3Agent:
    def __init__(self, k, gamma=0.1, rng=None):
        self.k = k
        self.gamma = gamma
        self.weights = np.ones(k)
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self):
	# adds up every arms' current wt into single number as per formula
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + (self.gamma / self.k)
        self.last_probs = probs
        return self.rng.choice(self.k, p=probs)

    def update(self, action, reward):
        estimated_reward = reward / self.last_probs[action]
        growth = np.exp((self.gamma * estimated_reward) / self.k)
        self.weights[action] *= growth
