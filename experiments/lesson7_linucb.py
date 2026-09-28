import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.ucb1 import UCB1Agent
from environments.contextual_bandit import ContextualBandit
from algorithms.linucb import LinUCBAgent

k = 10
d = 5
steps = 1000
runs = 500

def run_experiment():
    results = {"linucb": np.zeros(steps),"ucb1_no_context": np.zeros(steps), "random": np.zeros(steps)}
    master_rng = np.random.default_rng(2024)

    for run in range(runs):
        seed = master_rng.integers(1_000_000_000)
        bandit = ContextualBandit(k=k, d=d, seed=seed)
        agent = LinUCBAgent(k=k, d=d, alpha=1.0)
        ucb_rng = np.random.default_rng(master_rng.integers(1_000_000_000))
        ucb_agent = UCB1Agent(k=k, rng=ucb_rng)
        rand_rng = np.random.default_rng(master_rng.integers(1_000_000_000))

        for t in range(steps):
            ctx = bandit.new_context()
            best = bandit.optimal_action

            a = agent.select_action(ctx)
            r = bandit.pull(a)
            agent.update(a, ctx, r)
            if a == best:
                results["linucb"][t] += 1
            a_ucb = ucb_agent.select_action()
            r_ucb = bandit.pull(a_ucb)
            ucb_agent.update(a_ucb, r_ucb)
            if a_ucb == best:
                results["ucb1_no_context"][t] += 1
            if rand_rng.integers(k) == best:
                results["random"][t] += 1

    for key in results:
        results[key] = results[key] / runs * 100.0
    return results

if __name__ == "__main__":
    res = run_experiment()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "lesson7.npz")
    np.savez(save_path, **res)
    for name in res:
        print(f"{name:8s}  final (last 100 steps) % optimal = {res[name][-100:].mean():.2f}%")
