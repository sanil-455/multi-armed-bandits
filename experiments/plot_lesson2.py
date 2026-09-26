import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("results/lesson2.npz")

plt.figure(figsize=(9, 5))
plt.plot(data["optimistic_greedy"], color="blue", label="optimistic greedy (Q1=5, eps=0, alpha=0.1)")
plt.plot(data["realistic_eps_greedy"], color="gray", label="realistic eps-greedy (Q1=0, eps=0.1, alpha=0.1)")
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.title("Lesson 2: Optimistic Initial Values vs Realistic eps-greedy (S&B Fig 2.3 protocol)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/lesson2_plot.png", dpi=130)
print("saved to results/lesson2_plot.png")
