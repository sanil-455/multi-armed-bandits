import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.bernoulli_bandit import BernoulliBandit
from algorithms.ucb1 import UCB1Agent
from algorithms.thompson_sampling import ThompsonSamplingAgent

k = 10
steps = 1000
runs = 2000
# ucb1 fits in with bernolli as it says that rewards to be bounded in[0,1] and in nerolli its 
# already between [0,1]:: thus it can run in bernoulli env as well
AGENTS = {
    "ucb1":     lambda rng: UCB1Agent(k=k, rng=rng),
    "thompson": lambda rng: ThompsonSamplingAgent(k=k, rng=rng),
}

def run_experiment():
    results_optimal = {name: np.zeros(steps) for name in AGENTS}
    master_rng = np.random.default_rng(99)

    for name, make_agent in AGENTS.items():
        for run in range(runs):
            bandit_seed = master_rng.integers(1_000_000_000)
            bandit = BernoulliBandit(k=k, seed=bandit_seed)
            agent_rng = np.random.default_rng(master_rng.integers(1_000_000_000))
            agent = make_agent(agent_rng)

            for t in range( steps):
                a = agent.select_action()
                r = bandit.pull(a)
                agent.update(a, r)
                if a == bandit.optimal_action:
                    results_optimal[name][t] += 1

        results_optimal[name] = results_optimal[name] / runs * 100.0

    return results_optimal

if __name__ == "__main__":
    optimal_pct = run_experiment()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "lesson4.npz")
    np.savez(save_path, **optimal_pct)

    for name in AGENTS:
        print(f"{name:10s}  final (last 100 steps) % optimal = {optimal_pct[name][-100:].mean():.2f}%")
