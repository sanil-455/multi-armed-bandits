 ## Multi-Armed Bandits 
 
## Progression


## Greedy & ε-greedy

**Theory source:** Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 2.

**Cross-checked against:** [bgalbraith/bandits](https://github.com/bgalbraith/bandits)

**Files:**
- `environments/stationary.py` — k-armed Gaussian bandit testbed (S&B §2.3 "10-armed testbed")
- `algorithms/epsilon_greedy.py` — incremental sample-average estimator + ε-greedy action selection
- `experiments/lesson1_greedy_vs_epsilon.py` — reproduces S&B Fig 2.2 (2000 runs × 1000 steps)

**Result:** ε=0.1 reaches ~80% optimal action by step 1000; greedy (ε=0) plateaus at ~35% and never
improves further because it has no mechanism to revisit a bad early commitment. See `results/lesson1_plot.png`.

## Optimistic Initial Values

## Sutton & Barto, Chapter 2, Section 2.6.
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

## UCB1

Auer, Cesa-Bianchi and Fischer, Finite-time Analysis of the Multiarmed Bandit Problem,
Machine Learning 47, 2002. https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf

Referred: SMPyBandits' UCB.computeAllIndex() -- identical formula with the coefficient fixed
at sqrt(2), confirming this is the paper's exact UCB1 and not Sutton & Barto's more general
tunable-c variant (S&B eq 2.10 uses Q(a) + c*sqrt(ln t / N(a)) with c left free).

Main: every arm is played once before the confidence-bound formula is used at all, exactly as
the paper states, to avoid dividing by zero when N(a) = 0.

Result (2000 runs x 1000 steps, k=10)
- eps-greedy (eps=0.1): 80.03%, optimistic greedy: 87.21%, UCB1: 90.31% optimal by step 1000
- UCB1 wins because its exploration is aimed at genuinely uncertain arms and never fully
switches off, unlike eps-greedy's permanent uniform randomness or optimistic-greedy's
one-time decaying trick. See results/lesson3_plot.png.

## Thompson Sampling

Thompson, W.R. (1933), On the likelihood that one unknown probability exceeds another in view
of the evidence of two samples, Biometrika 25(3/4), 285-294.
https://www.gwern.net/doc/statistics/decision/1933-thompson.pdf

Russo, Van Roy, Kazerouni, Osband, Wen (2018), A Tutorial on Thompson Sampling, Foundations
and Trends in Machine Learning 11(1). https://arxiv.org/abs/1707.02038

Main: new BernoulliBandit environment, since Thompson Sampling needs binary rewards unlike the
Gaussian environment used earlier. Every arm starts at Beta(1,1), the uniform distribution,
meaning no prior belief at all. After each pull alpha increases on a success and beta on a
failure -- the standard Beta-Bernoulli conjugate update. Instead of one point estimate per arm
it keeps a whole distribution, samples one value from each arm's distribution every round, and
picks whichever sample came out highest, so arms it is still uncertain about get explored
naturally.

Result (2000 runs x 1000 steps, k=10, Bernoulli bandit)
- UCB1: 55.17%, Thompson Sampling: 87.39% optimal action by step 1000
- Matches a known published result: Chapelle and Li (2011, NeurIPS), An Empirical Evaluation
of Thompson Sampling, found the same thing on Bernoulli payoffs -- UCB1's theoretical constant
is proven-safe but loose, so it explores more conservatively than it needs to.
See results/lesson4_plot.png.

## Softmax / Boltzmann Exploration

Cesa-Bianchi, Gentile, Lugosi, Neu (2017), Boltzmann Exploration Done Right, NeurIPS 30.
https://arxiv.org/abs/1705.10257

The paper shows the usual way people implement Boltzmann exploration, with one shared learning
rate for all arms, is basically no better than eps-greedy -- any schedule either gets stuck or
explores forever. Their fix is Boltzmann-Gumbel Exploration: give each arm its own random
Gumbel noise, scaled down as that arm gets pulled more, added on top of its plain average.

Result (2000 runs x 1000 steps, k=10, Gaussian bandit)
- UCB1: 90.31%, BGE: 85.94%, eps-greedy: 80.03% optimal action by step 1000
- BGE coming in behind UCB1 is what the paper's own maths predicts: its regret bound carries an
extra log T factor compared to UCB1's.

Also tried building the heavier Theorem 5 version (a Catoni-based estimator meant to resist
outlier rewards) and testing it on a heavy-tailed environment. It did not come out clearly
better than plain BGE or UCB1 in this run -- most likely needs more runs and a better-tuned
constant to show the effect. See results/lesson5_plot.png.

## EXP3

Auer, Cesa-Bianchi, Freund, Schapire (2002), The Nonstochastic Multiarmed Bandit Problem,
SIAM Journal on Computing 32(1), 48-77.

No assumption at all about how rewards are generated -- built for a setting where an adversary
controls the payoffs. Keeps a weight per arm instead of a value estimate, mixes in a flat
gamma/K of forced exploration on top of the weighted probabilities, and corrects for only
seeing one arm's reward per round by dividing by the probability that arm had of being picked.

Referred: the SCIP solver's own EXP3 implementation -- the probability formula matches exactly.
Noticed a difference in the weight-update exponent (they use a fixed 1/K, we use gamma/K per
the classic formula) -- worth digging into further, not resolved yet.

New environment: environments/non_stationary_bandit.py, where the best arm secretly switches
partway through the run (step 500 of 1000).

Result (2000 runs x 1000 steps, k=10, change at step 500)
- UCB1: 85.69% just before the change, 50.54% average after
- EXP3: 66.46% just before the change, 25.80% average after
- EXP3 did worse, not better. Checked this was not a bug -- tested bounded rewards and swept
gamma including the paper's own theoretically optimal value (0.164 for these settings), which
only reached 69.82%. So it is real: EXP3's sqrt(T) bound is genuinely weaker in absolute terms
than UCB1's log T bound at this horizon, UCB1 already re-explores somewhat since its confidence
term keeps growing with total time, and a single scheduled change point is not the reactive
adversarial setting EXP3 is actually built to guarantee against. See results/lesson6_plot.png.

## LinUCB

Li, Chu, Langford, Schapire (2010), A Contextual-Bandit Approach to Personalized News Article
Recommendation, WWW 2010. https://arxiv.org/abs/1003.0146

First algorithm here that gets to see a context vector before choosing. Assumes reward is
linear in the context features, keeps a ridge regression per arm (A starts as the d x d
identity, b as a zero vector), and picks the arm maximising predicted reward plus a confidence
width alpha * sqrt(x' A^-1 x) -- the same estimate-plus-uncertainty shape as UCB1, but the
uncertainty is now measured in feature space rather than just a pull count.

Referred: the contextual R package's LinUCBDisjointPolicy docs, which confirm the same
initialisation (A = d x d identity, b = zero vector of length d, alpha as the exploration
hyperparameter).

New environment: environments/contextual_bandit.py, where each arm has its own secret
coefficient vector and a fresh random context is drawn every round, so the best arm changes
constantly.

Result (500 runs x 1000 steps, k=10, d=5)
- LinUCB: 96.45%, context-blind UCB1: 8.20%, random baseline: 10.08%
- UCB1 lands below random, which makes sense: it converges on whichever arm looks best averaged
over all contexts and then commits to it, and committing to one fixed answer is worse than
guessing when the right answer changes every round. Clearest demonstration in this repo of why
contextual bandits needed to exist. See results/lesson7_plot.png.
