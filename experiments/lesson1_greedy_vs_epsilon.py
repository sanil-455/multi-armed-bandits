"""
Lesson 1 experiment.

We run three agents: greedy (eps=0), eps=0.01, eps=0.1
across 2000 independent bandit problems, 1000 steps each,
and average the results. This IS the S&B Fig 2.2 protocol
(same numbers: 2000 runs, 1000 steps, k=10).

Why 2000 independent runs? Because ONE run is noisy -- a single greedy
run might get lucky. We need to average over many random problem instances
to see the TRUE expected behavior of the algorithm, not one lucky/unlucky draw.
This is basic Monte Carlo experimental methodology and it's why every
bandit paper reports results averaged over many runs.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.stationary import StationaryBandit
from algorithms.epsilon_greedy import EpsilonGreedyAgent

K = 10
STEPS = 1000
RUNS = 2000
EPSILONS = [0.0, 0.01, 0.1]

def run_experiment():
    results_reward = {eps: np.zeros(STEPS) for eps in EPSILONS}
    results_optimal = {eps: np.zeros(STEPS) for eps in EPSILONS}

    master_rng = np.random.default_rng(42)

    for eps in EPSILONS:
        for run in range(RUNS):
            bandit_seed = master_rng.integers(1_000_000_000)
            bandit = StationaryBandit(k=K, seed=bandit_seed)
            agent_rng = np.random.default_rng(master_rng.integers(1_000_000_000))
            agent = EpsilonGreedyAgent(k=K, epsilon=eps, rng=agent_rng)

            for t in range(STEPS):
                a = agent.select_action()
                r = bandit.pull(a)
                agent.update(a, r)
                results_reward[eps][t] += r
                if a == bandit.optimal_action:
                    results_optimal[eps][t] += 1

        results_reward[eps] /= RUNS
        results_optimal[eps] = results_optimal[eps] / RUNS * 100.0

    return results_reward, results_optimal

if __name__ == "__main__":
    rewards, optimal_pct = run_experiment()
    np.savez("/home/claude/mab_course/results/lesson1.npz",
             **{f"reward_eps{eps}": rewards[eps] for eps in EPSILONS},
             **{f"optimal_eps{eps}": optimal_pct[eps] for eps in EPSILONS})

    # Print some real, honest numbers -- not invented ones
    for eps in EPSILONS:
        print(f"eps={eps:5.2f}  final avg reward (last 100 steps) = {rewards[eps][-100:].mean():.4f}"
              f"   final % optimal action (last 100 steps) = {optimal_pct[eps][-100:].mean():.2f}%")
