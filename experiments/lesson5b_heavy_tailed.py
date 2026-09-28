import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.heavy_tailed_bandit import HeavyTailedBandit
from algorithms.ucb1 import UCB1Agent
from algorithms.boltzmann_gumbel import BGEAgent
from algorithms.robust_boltzmann_gumbel import RobustBGEAgent

k = 10
steps = 500
runs = 500

AGENTS = {
    "ucb1":       lambda rng: UCB1Agent(k=k, rng=rng),
    "bge":        lambda rng: BGEAgent(k=k, C=1.0, rng=rng),
    "robust_bge": lambda rng: RobustBGEAgent(k=k, C=1.0, c=1.0, rng=rng),
}
def run_experiment():
    results_optimal = {name: np.zeros(steps) for name in AGENTS}
    master_rng = np.random.default_rng(31)

    for name, make_agent in AGENTS.items():
        for run in range(runs):
            bandit_seed = master_rng.integers(1_000_000_000)
            bandit = HeavyTailedBandit(k=k, df=3, seed=bandit_seed)
            agent_rng = np.random.default_rng(master_rng.integers(1_000_000_000))
            agent = make_agent(agent_rng)

            for t in range(steps):
                a = agent.select_action()
                r = bandit.pull(a)
                agent.update(a, r)
                if a == bandit.optimal_action:
                    results_optimal[name][t] += 1

        results_optimal[name] = results_optimal[name] / runs * 100.0

    return results_optimal

if __name__ == "__main__":
    optimal_pct = run_experiment()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "lesson5b.npz")
    np.savez(save_path, **optimal_pct)

    for name in AGENTS:
        print(f"{name:12s}  final (last 50 steps) % optimal = {optimal_pct[name][-50:].mean():.2f}%")

