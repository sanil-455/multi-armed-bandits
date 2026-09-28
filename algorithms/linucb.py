import numpy as np

class LinUCBAgent:
    def __init__(self, k, d, alpha=1.0):
        self.k = k
        self.d = d
        self.alpha = alpha
        self.A = np.array([np.identity(d) for _ in range(k)])
        self.b = np.zeros((k, d))

    def select_action(self, context):
        scores = np.zeros(self.k)
        for a in range(self.k):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            mean = theta_hat @ context
            bonus = self.alpha * np.sqrt(context @ A_inv @ context)
            scores[a] = mean + bonus
        return int(np.argmax(scores))

    def update(self, action, context, reward):
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
