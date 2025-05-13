# Gridworld via Dynamic Programming

This repository demonstrates the application of **Dynamic Programming** to solve a **Gridworld** task using **Policy Evaluation**, **Policy Improvement**, and **Policy Iteration**. The task consists of a 4x4 grid, and the agent must learn optimal policies through iterative computation of state-value functions.

## Background

In this example:

* **Gridworld** consists of 14 non-terminal states and a terminal state.
* **Actions**: The agent can perform 4 actions in each state: **up**, **down**, **right**, and **left**.
* The **reward** for all transitions is **-1** until the terminal state is reached, where the reward is **+1**.
* The task is **episodic** and **undiscounted**.

### Policy Evaluation (Prediction)

Policy Evaluation computes the state-value function for a given policy by iterating over the gridworld. The value of each state represents the expected number of steps from that state to the terminal state.

---

## Structure

* `grid_world.py`: Contains functions for the **state transitions**, **reward calculation**, and **value function computation**.
* `notebooks`: This file contains the policy evaluation, policy improvement, and policy iteration logic.
* `generated_images`: Directory where result plots are saved.

---

## Simulation Overview

The simulation involves:

* **State transitions**: Deterministic, except when an action would cause the agent to go off the grid.
* **Policy Evaluation**: Computes the state-value function using iterative updates until convergence.
* **Policy Improvement**: Updates the policy based on the computed value function.
* **Policy Iteration**: Computes the optimal policy in a few iterations.

### Figures & Their Meanings

#### Figure 4.1 —  In-Place Policy Evaluation

![figure_4_1_in_place.png](gridworld-dp/generated_images/figure_4_1_in_place.png)

This figure shows the value function convergence over several iterations. On the left, the sequence of approximations of the value function for a random policy is shown. On the right, the corresponding greedy policies are illustrated.

Purpose: This demonstrates how iterative policy evaluation converges to the true state-value function and how the policy improves after each iteration.



#### Figure 4.2 — Out-of-Place Policy Evaluation

![figure_4_1_out_place.png](gridworld-dp/generated_images/figure_4_1_out_place.png)

This figure illustrates the value function convergence for an out-of-place computation method, which computes the state values without modifying the existing value function during the process. The sequence of approximations of the value function is shown on the left for a random policy, while the corresponding greedy policies are shown on the right.

Purpose: To highlight the difference in convergence behavior when performing the value function evaluation without modifying the current state-value table in-place. This shows that both in-place and out-of-place methods lead to improved policies, with the out-of-place method providing a more stable progression of value function updates.

---

## Policy Iteration

Policy Iteration is an alternative approach to finding the optimal policy. It involves two steps: policy evaluation and policy improvement, which are repeated until the policy stabilizes.

In this example, policy iteration converges in just one iteration, as the initial policy already leads to the optimal solution.

Purpose: To illustrate how policy iteration can efficiently converge to the optimal policy in dynamic programming tasks

---

## Reference

Sutton, R. S., & Barto, A. G. (2018).
**Reinforcement Learning: An Introduction (2nd Ed.)**

[📘 Free PDF](http://incompleteideas.net/book/the-book.html)

---

## Educational Objective

This project is designed for educational purposes, specifically for students studying reinforcement learning. It aims to build intuition about **policy evaluation**, **policy improvement**, and **policy iteration** in a simple yet rich environment, illustrating key concepts in **dynamic programming**.


