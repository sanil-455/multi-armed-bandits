import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("results/lesson7.npz")

plt.figure(figsize=(9, 5))
plt.plot(data["linucb"], color="green", label="LinUCB (sees context)")
plt.plot(data["ucb1_no_context"], color="red", label="UCB1 (cannot see context)")
plt.plot(data["random"], color="gray", linestyle="--", label="random baseline")
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.title("Lesson 7: LinUCB vs context-blind UCB1 (500 runs, k=10, d=5)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/lesson7_plot.png", dpi=130)
print("saved")
