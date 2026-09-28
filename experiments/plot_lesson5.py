import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("results/lesson5.npz")

plt.figure(figsize=(9, 5))
plt.plot(data["eps_greedy_0.1"], color="gray", label="eps-greedy (eps=0.1)")
plt.plot(data["ucb1"], color="red", label="UCB1")
plt.plot(data["bge"], color="purple", label="Boltzmann-Gumbel Exploration")
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.title("Lesson 5: BGE vs UCB1 vs eps-greedy (2000 runs, k=10, Gaussian bandit)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/lesson5_plot.png", dpi=130)
print("saved to results/lesson5_plot.png")
