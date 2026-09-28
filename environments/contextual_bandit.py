import numpy as np

class ContextualBandit:
    def __init__(self, k=10, d=5, seed=None):
        self.k = k
        self.d = d
        self.rng = np.random.default_rng(seed)
        self.theta_star = self.rng.normal(loc=0.0, scale=1.0, size=(k, d))
        self.current_context = None

    def new_context(self):
        self.current_context = self.rng.normal(loc=0.0, scale=1.0, size=self.d)
        return self.current_context

    def pull(self, action):
        expected = self.theta_star[action] @ self.current_context
        return expected + self.rng.normal(0.0, 0.1)

    @property
    def optimal_action(self):
        expected_all = self.theta_star @ self.current_context
        return int(np.argmax(expected_all))
