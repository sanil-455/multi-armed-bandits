import numpy as np

class HeavyTailedBandit:
    def __init__(self, k=10, df=3, seed=None):
        self.k = k
        self.df = df # controls the shape of probabilty distribution (here 3)
        self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.normal(loc=0.0, scale=1.0, size=k) # this creates secret true values for each 
	# arm which is never disclosed to agent and agent only estimates through noisy onservations
        self.optimal_action = int(np.argmax(self.q_star)) # returns teh position of the largest value

    def pull(self, action): # action tells which arm we want to pull
	# self.rng.standard_t(self.df) draws one random number from a Student's t-distribution 
	# a Student's t-distribution looks similar to a Normal distribution near its center but it has fatter 
	# tails,ie. extreme far from centervalues
	# self .df controls how fat this tails will be:
	# small df means more fat tails hence more freq outliers while self.df at inf is normal distribution 
        noise = self.rng.standard_t(self.df)
	# self.q_star[action] looks up this specific arm's true average value. then we add random noise on it.
        return self.q_star[action] + noise
