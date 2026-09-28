import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("results/lesson6.npz")

plt.figure(figsize=(9, 5))
plt.plot(data["ucb1"], color="red", label="UCB1")
plt.plot(data["exp3"], color="orange", label="EXP3")
plt.axvline(x=500, color="black", linestyle="--", label="change point")
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.title("Lesson 6: EXP3 vs UCB1 across a non-stationary change (2000 runs, k=10)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/lesson6_plot.png", dpi=130)
print("saved")
