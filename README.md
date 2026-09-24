# Multi-Armed Bandits — From First Principles

Personal implementation + study repo for classical and contextual bandit algorithms,
built while studying for [research area/professor's lab].

Every algorithm here is implemented from its original paper/textbook chapter
(cited below), then cross-checked against at least one independent open-source
implementation to catch mistakes. No algorithm is copy-pasted from another repo.

## Progression

Greedy → ε-greedy → Optimistic Initialization → UCB1 → Thompson Sampling → Softmax → EXP3 → LinUCB

## Lesson 1: Greedy & ε-greedy — DONE

**Theory source:** Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 2.

**Cross-checked against:** [bgalbraith/bandits](https://github.com/bgalbraith/bandits)

**Files:**
- `environments/stationary.py` — k-armed Gaussian bandit testbed (S&B §2.3 "10-armed testbed")
- `algorithms/epsilon_greedy.py` — incremental sample-average estimator + ε-greedy action selection
- `experiments/lesson1_greedy_vs_epsilon.py` — reproduces S&B Fig 2.2 (2000 runs × 1000 steps)

**Result:** ε=0.1 reaches ~80% optimal action by step 1000; greedy (ε=0) plateaus at ~35% and never
improves further because it has no mechanism to revisit a bad early commitment. See `results/lesson1_plot.png`.

## Lesson 2: Optimistic Initial Values — NEXT
