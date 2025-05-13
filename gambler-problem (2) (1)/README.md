# Gambler’s Problem — Value Iteration

## Background

This notebook implements the **Gambler’s Problem** as introduced in Sutton & Barto's *Reinforcement Learning: An Introduction* (Example 4.3). In this episodic, undiscounted setting:

* A gambler bets on a biased coin flip (with head probability *ph*).
* The goal is to reach 100 dollars from an initial capital using optimal betting strategies.
* The state space S = {1, 2, ..., 99} represents capital.
* The action space is the amount staked on each bet: a ∈ {0, 1, ..., min(s, 100 − s)}.
* Reward is +1 for reaching 100, and 0 otherwise.

The problem is solved via **value iteration** to find the optimal value function and corresponding optimal policy.

---

## Structure

* **MDP Setup**: States, actions, transition probabilities, and rewards are defined for a biased coin (ph = 0.4).
* **Value Iteration Loop**: Iteratively updates the value function based on Bellman's optimality equation.
* **Policy Extraction**: At each step, the best action (stake) is selected greedily based on expected return.
* **Convergence**: Iteration halts when the value function changes below a threshold.

---

## Simulation Overview

* Initialization:

  * Terminal state value set to 1 at goal (100 dollars).
  * All other values initialized to 0.
* Loop continues until value estimates converge.
* At each iteration:

  * For every state *s*, evaluate all valid actions *a*.
  * Compute the expected value based on outcomes (win or lose).
  * Update V(s) and record policy π(s).

---

## Figures with Interpretations

![img.png](img.png)

**Figure 4.3 — Solution to the Gambler’s Problem for ph = 0.4**

* **Top Plot (Value Function)**:

  * Shows how the value estimates evolve over successive sweeps.
  * Converges to the optimal value function *v\**.

* **Bottom Plot (Final Policy)**:

  * Displays the final optimal stake (action) for each capital level.
  * Indicates that optimal strategies may involve **aggressive** betting, especially near 100.
  * Multiple optimal policies may exist, especially near the goal, due to tied argmax values.

---

## Reference

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Example 4.3.

---

## Educational Objective

* Illustrate the use of **value iteration** in an episodic, finite MDP setting.
* Highlight how optimal policies can emerge from **expected return maximization**.
* Provide intuition on how **risk vs. reward** plays out in stochastic environments.


