"""
Stationary k-armed bandit testbed.

This follows Sutton & Barto (2nd ed.), Section 2.3, "The 10-armed Testbed" exactly:
- q*(a) for each of the k arms is drawn once from a Normal(0, 1) distribution.
- Pulling arm a returns a reward drawn from Normal(q*(a), 1).
- "Stationary" means q*(a) does NOT change over time. (We'll build a non-stationary
  version later, in Lesson 8, for EXP3.)
"""
import numpy as np


class StationaryBandit:
    def __init__(self, k=10, seed=None):
        self.k = k
        self.rng = np.random.default_rng(seed)
        # true action values q*(a), a=1..k -- UNKNOWN to any agent, only used
        # internally to generate rewards and to score optimality.
        self.q_star = self.rng.normal(loc=0.0, scale=1.0, size=k)
        self.optimal_action = int(np.argmax(self.q_star))

    def pull(self, action):
        """Return a single sampled reward for pulling `action`."""
        return self.rng.normal(loc=self.q_star[action], scale=1.0)

    def reset_rewards_only(self):
        """Reset noise stream but keep the same q_star (same problem instance)."""
        pass  # rewards are stochastic per pull() call, nothing to reset
