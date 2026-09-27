import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.stationary import StationaryBandit
from algorithms.epsilon_greedy import EpsilonGreedyAgent
from algorithms.ucb1 import UCB1Agent

# comparing UCB and eps-greedy on same terms
k = 10
steps = 1000
runs = 2000

from algorithms.optimistic_initialization import OptimisticGreedyAgent
# a diffrence from lesson 2 we did here is that the indices in given dict comes from diff classes 
# we created a lamda fn instaed of normal for these as rng value will be fixed later
# if we created like EpsilonGreedyAgent(...) directly, we would've to give rng immediately
AGENTS = {
    "eps_greedy_0.1":    lambda rng: EpsilonGreedyAgent(k=k, epsilon=0.1, rng=rng),
    "optimistic_greedy": lambda rng: OptimisticGreedyAgent(k=k, initial_value=5.0, epsilon=0.0, step_size=0.1, rng=rng),
    "ucb1":              lambda rng: UCB1Agent(k=k, rng=rng),
}
def run_experiment():
    results_optimal = {name: np.zeros(steps) for name in AGENTS}
    master_rng = np.random.default_rng(7)

    for name, make_agent in AGENTS.items():
        for run in range(runs):
            bandit_seed = master_rng.integers(1_000_000_000)
            bandit = StationaryBandit(k=k, seed=bandit_seed)
            agent_rng = np.random.default_rng(master_rng.integers(1_000_000_000))
		# only change from prev code is that here we call make_agent from AGENT dict instead
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
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "lesson3.npz")
    np.savez(save_path, **optimal_pct)

    for name in AGENTS:
        print(f"{name:20s}  final (last 100 steps) % optimal = {optimal_pct[name][-100:].mean():.2f}%")

