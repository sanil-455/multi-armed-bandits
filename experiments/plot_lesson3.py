import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("results/lesson3.npz")

plt.figure(figsize=(9, 5))
plt.plot(data["eps_greedy_0.1"], color="gray", label="eps-greedy (eps=0.1)")
plt.plot(data["optimistic_greedy"], color="blue", label="optimistic greedy (Q1=5, alpha=0.1)")
plt.plot(data["ucb1"], color="red", label="UCB1 (Auer et al. 2002)")
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.title("Lesson 3: UCB1 vs eps-greedy vs optimistic-greedy (2000 runs, k=10)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/lesson3_plot.png", dpi=130)
print("saved to results/lesson3_plot.png")
