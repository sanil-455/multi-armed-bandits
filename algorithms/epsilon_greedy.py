"""
Greedy and epsilon-greedy action-selection.

Both use the SAME incremental sample-average update (Sutton & Barto eq. 2.3):
    Q(a) <- Q(a) + (1/N(a)) * [R - Q(a)]

The ONLY difference between "greedy" and "epsilon-greedy" is what select_action()
does. This is deliberate -- I want you to see that these are not two unrelated
algorithms, they're the same estimator with two different action-selection rules
bolted on. That's true of almost every classical bandit algorithm: separate the
"how do I estimate Q" question from "how do I choose an action given Q".
"""
import numpy as np


class EpsilonGreedyAgent:
    def __init__(self, k, epsilon=0.0, initial_value=0.0, rng=None):
        """
        epsilon=0.0  -> this IS pure greedy (a special case, not separate code)
        epsilon=0.1  -> 10% of the time, explore uniformly at random
        """
        self.k = k
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()
        self.Q = np.full(k, float(initial_value))   # current value estimates
        self.N = np.zeros(k, dtype=int)              # pull counts per arm

    def select_action(self):
        if self.rng.random() < self.epsilon:
            # EXPLORE: uniform random arm, including possibly the current best
            return self.rng.integers(self.k)
        else:
            # EXPLOIT: pick arm with highest current estimate.
            # Ties broken randomly -- if you don't do this, in early steps
            # (all Q=0) you ALWAYS pick arm 0, which silently breaks the
            # algorithm. This is a real bug people ship.
            max_q = np.max(self.Q)
            candidates = np.flatnonzero(self.Q == max_q)
            return self.rng.choice(candidates)

    def update(self, action, reward):
        self.N[action] += 1
        # incremental sample-average update, S&B eq. 2.3
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])
