import numpy as np

def catoni_psi(x):
    if x >= 0:
        return np.log(1 + x + (x**2) /2)
    else:
        return -np.log(1 - x + (x**2) /2)

class RobustBGEAgent:
    def __init__(self, k, C=1.0, c=1.0, rng=None):
        self.k = k
        self.C = C
        self.c = c
        self.N = np.zeros(k, dtype=int)
        self.rewards = [[] for _ in range(k)]
        self.rng = rng if rng is not None else np.random.default_rng()

    def _robust_mean(self, arm):
        n = self.N[arm]
        beta = np.sqrt((2.0 * np.log(1.0 / 0.05)) / (n * self.c ** 2))
        total = 0.0
        for r in self.rewards[arm]:
            total += catoni_psi(beta * r)
        return total / (n * beta)

    def select_action(self):
        untried = np.flatnonzero(self.N == 0)
        if len(untried) > 0:
            return untried[0]

        means = np.array([self._robust_mean(i) for i in range(self.k)])
        explore_beta = np.sqrt((self.C ** 2) / self.N)
        Z = self.rng.gumbel(size=self.k)
        perturbed = means + explore_beta * Z
        return int(np.argmax(perturbed))

    def update(self, action, reward):
        self.N[action] += 1
        self.rewards[action].append(reward)
