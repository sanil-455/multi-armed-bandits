import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.non_stationary_bandit import NonStationaryBandit
from algorithms.ucb1 import UCB1Agent
from algorithms.exp3 import EXP3Agent

k = 10
steps = 1000
runs = 2000
change_point = 500

AGENTS = {
    "ucb1": lambda rng: UCB1Agent(k=k, rng=rng),
    "exp3": lambda rng: EXP3Agent(k=k, gamma=0.1, rng=rng),
}

def run_experiment():
    results_optimal = {name: np.zeros(steps) for name in AGENTS}
    master_rng = np.random.default_rng(77)

    for name, make_agent in AGENTS.items():
        for run in range(runs):
            bandit_seed = master_rng.integers(1_000_000_000)
            bandit = NonStationaryBandit(k=k, change_point=change_point, seed=bandit_seed)
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
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "lesson6.npz")
    np.savez(save_path, **optimal_pct)

    for name in AGENTS:
        before = optimal_pct[name][:change_point][-100:].mean()
        after = optimal_pct[name][change_point:].mean()
        print(f"{name}: last-100-before-change = {before:.2f}%, avg-after-change = {after:.2f}%")
