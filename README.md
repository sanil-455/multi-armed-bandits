# Multi-Armed Bandits — From First Principles

Personal implementation + study repo for classical and contextual bandit algorithms,
built while studying for [research area/professor's lab].

Every algorithm here is implemented from its original paper/textbook chapter
(cited below), then cross-checked against at least one independent open-source
implementation to catch mistakes. No algorithm is copy-pasted from another repo.

## Progression

Greedy → ε-greedy → Optimistic Initialization → UCB1 → Thompson Sampling → Softmax → EXP3 → LinUCB

 Greedy & ε-greedy

**Theory source:** Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 2.

**Cross-checked against:** [bgalbraith/bandits](https://github.com/bgalbraith/bandits)

**Files:**
- `environments/stationary.py` — k-armed Gaussian bandit testbed (S&B §2.3 "10-armed testbed")
- `algorithms/epsilon_greedy.py` — incremental sample-average estimator + ε-greedy action selection
- `experiments/lesson1_greedy_vs_epsilon.py` — reproduces S&B Fig 2.2 (2000 runs × 1000 steps)

**Result:** ε=0.1 reaches ~80% optimal action by step 1000; greedy (ε=0) plateaus at ~35% and never
improves further because it has no mechanism to revisit a bad early commitment. See `results/lesson1_plot.png`.

Optimistic Initial Values

Sutton & Barto, Chapter 2, Section 2.6.
Referred: kamenbliznashki/sutton_barto (fig_2_3() in ch02_ten_armed_testbed.py)
identical configuration (Q1=[5,0], eps=[0,0.1], step_size=0.1) and identical constant-step-size
the update rule has been confirmed independently by reading the actual source code.

Main: uses constant step-size alpha=0.1, NOT the 1/N sample-average from
1st chapter -- sample-averaging would erase the optimistic initial value after just one pull of each arm,
since step-size = 1/N = 1 on the very first pull.

Result (2000 runs x 1000 steps, k=10)
- optimistic greedy (Q1=5, eps=0): 13.14% optimal action in first 20 steps, 86.09% by step 1000
- realistic eps-greedy (Q1=0, eps=0.1): 25.79% optimal action in first 20 steps, 75.59% by step 1000
- Optimistic starts worse (still disappointment-driven exploring) but ends better, since it stops
paying an ongoing exploration cost once its optimism decays away; eps-greedy pays a permanent 10%-random
exploration tax forever.
