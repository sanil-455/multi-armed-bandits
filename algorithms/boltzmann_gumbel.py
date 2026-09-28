import numpy as np

class BGEAgent:
    def __init__(self, k, C=1.0, rng=None):
        self.k = k
        self.C = C
        self.N = np.zeros(k, dtype=int) # pulls per arm
        self.Q = np.zeros(k) # plain sample-average estimates
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self):
	# checks if every arm is tried once
        untried = np.flatnonzero(self.N == 0)
        if len(untried) > 0:
            return untried[0]

        beta = np.sqrt((self.C ** 2) / self.N) # self.N is the whole array
        Z = self.rng.gumbel(size=self.k) # draws 1 independent gumbel distributed number per arm all at once
					# self,k tells numpy to give k seperate draws producing an array
					# this is what makes it random rather than deterministic
        perturbed = self.Q + beta * Z # formula in theory
				# his scales each arm's random noise by that arm's own current uncertaintya
				# a heavily-tried arm has small beta, so even a large random Z draw barely
				# moves its perturbed value; a rarely-tried arm has large beta, so its random
				# draw can swing its perturbed value substantially.
        return int(np.argmax(perturbed)) # pick whichever arm has the highest perturbed (noise-added) value now

    def update(self, action, reward):
	#this is the exact same 1/N sample-average
	# increases pulled armscount by 1x
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])
