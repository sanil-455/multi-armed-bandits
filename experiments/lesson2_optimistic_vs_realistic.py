import numpy as np
# sys to interact with the python interpretor,os with operating system
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.stationary import StationaryBandit
from algorithms.optimistic_initialization import OptimisticGreedyAgent
# no. of arms per bandit problem
k = 10
# how many pulls each single simulated agent gets per run
steps = 1000
# no. of independent seprate bandit problems we will simulate and average over
runs = 2000
# constant step size
alpha = 0.1
# different setting to be tried
CONFIGS = {
    "optimistic_greedy":    dict(initial_value=5.0, epsilon=0.0, step_size=alpha),
    "realistic_eps_greedy": dict(initial_value=0.0, epsilon=0.1, step_size=alpha),
}
def run_experiment():
    # we end up with 2 arrays ie. for those in config dict each 1000 slots long total of 2000 times
    # thus we accumulate counts of how many times we pick the optimal arm of 2000 runs
    results_optimal = {name: np.zeros(steps) for name in CONFIGS}
    # using the same seed for generating the same 2000 bandit problem and same results every time
    master_rng = np.random.default_rng(123)

    for name, cfg in CONFIGS.items():
        for run in range(runs):
        # pull one number from master generator ise use for specific bandits run
            bandit_seed = master_rng.integers(1_000_000_000) 
        # creates a fresh 10 arm bandit problem using that seed
        # as the seed is different every run each of the 2000 runs gets a different problem
            bandit = StationaryBandit(k=k, seed=bandit_seed)
        # creating a seperate random generator differnt from bandit own randomness
        #  helps to distingiush between which bandit prob and which random exploration agentdoes
            agent_rng = np.random.default_rng(master_rng.integers(1_000_000_000))
        # create a fresh agent for the run
            agent = OptimisticGreedyAgent(k=k, rng=agent_rng, **cfg)

        #innermost loop for 100 individual pulls
            for t in range(steps):
        # asking the agent to choose an arm
                a = agent.select_action()
        # actually pull an arm and get a reward
                r = bandit.pull(a)
        # updating teh internal belief Q[a] using const alpha formula
                agent.update(a, r)
        # check if the one we picked is the best arm
                if a == bandit.optimal_action:
        # if yes increment by 1
                    results_optimal[name][t] += 1

        results_optimal[name] = results_optimal[name] / runs * 100.0
    # handing the final dict containing the percentage curve
    return results_optimal
if __name__ == "__main__":
    optimal_pct = run_experiment()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "lesson2.npz")
    np.savez(save_path, **optimal_pct)

    for name in CONFIGS:
        print(f"{name:22s}  first-20-step avg % optimal = {optimal_pct[name][:20].mean():.2f}%"
              f"   final (last 100 steps) % optimal = {optimal_pct[name][-100:].mean():.2f}%")
