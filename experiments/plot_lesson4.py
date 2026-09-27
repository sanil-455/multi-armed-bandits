import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("results/lesson4.npz")

plt.figure(figsize=(9, 5))
plt.plot(data["ucb1"], color="red", label="UCB1")
plt.plot(data["thompson"], color="green", label="Thompson Sampling")
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.title("Lesson 4: Thompson Sampling vs UCB1 on Bernoulli bandit (2000 runs, k=10)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/lesson4_plot.png", dpi=130)
print("saved to results/lesson4_plot.png")
