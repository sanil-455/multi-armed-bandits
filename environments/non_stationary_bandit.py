import numpy as np

class NonStationaryBandit:
	# change point is the pt at which value of best arm(mostly) changes
    def __init__(self, k=10, change_point=500, seed=None):
        self.k = k
        self.change_point = change_point
        self.rng = np.random.default_rng(seed)
	# set of true arm values for the 2 parts of the run
        self.q_star_before = self.rng.normal(loc=0.0, scale=1.0, size=k)
        self.q_star_after = self.rng.normal(loc=0.0, scale=1.0,size=k)
        self.t = 0

    def pull(self, action):
	# every single time an arm is pulled it increases
        self.t+= 1
        current_q_star = self.q_star_before if self.t <= self.change_point else self.q_star_after
        return self.rng.normal(loc=current_q_star[action], scale=1.0)

    @property
    def optimal_action(self):
        current_q_star = self.q_star_before if self.t <= self.change_point else self.q_star_after
        return int(np.argmax(current_q_star))
