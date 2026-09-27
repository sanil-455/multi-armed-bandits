import numpy as np


class BernoulliBandit:
    def __init__(self, k=10, seed=None):
        self.k = k
        self.rng = np.random.default_rng(seed)
	# instead of drawing Gaussian true values (q-star, which could be any real number),
	# we choose uniform random numbers between 0 and 1 — one per arm.Each of these represents 
	# that arm's true success probability — e.g., p_star = [0.73, 0.12, 0.55, ...] means arm 
	# 0 secretly 73% of the time, arm 1 only 12% of the time so on
        self.p_star = self.rng.uniform(low=0.0, high=1.0, size=k)
	# find which arm has highest true success rate,store it for later scoring(agent doesnt know)
        self.optimal_action = int(np.argmax(self.p_star))
	
#  simulate pulling 1 arm
    def pull(self, action):
	# self.rng.random returns a no. randomly and comapring it with p.star(action) 
	# if arm's success rate is 0.28 for eg: then 28% of time we get true,72% of time false
	# and thus we get 1- true and 0- false at the freq of binomial distribution
        return int(self.rng.random() < self.p_star[action])
