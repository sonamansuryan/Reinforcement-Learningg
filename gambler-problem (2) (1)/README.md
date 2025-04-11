# Gambler's Problem – Value Iteration

This project provides an implementation of the **Gambler’s Problem**, as described in Sutton and Barto's *Reinforcement Learning: An Introduction (2nd Edition)*, Chapter 4. It uses **value iteration** to determine the **optimal policy** and **state-value function** for the gambler, where the objective is to reach a goal of 100 dollars through bets on a biased coin flip.

## Problem Description

- The gambler bets on a coin flip with a known probability of heads, `p_h`.
- The game ends either when the gambler’s capital reaches $100 (win) or drops to $0 (loss).
- At each step:
  - The gambler chooses a stake (an integer amount of current capital).
  - If heads: he gains that amount; if tails: he loses it.
- The goal is to maximize the probability of reaching $100.

## Markov Decision Process (MDP) Formulation

- **States**: Capital levels from `1` to `99` (0 and 100 are terminal).
- **Actions**: Stakes from `1` to `min(state, 100 - state)`.
- **Reward**: `+1` on reaching 100, `0` otherwise.
- **Transition probabilities**: Determined by `p_h`.

## Method: Value Iteration

- Iteratively update state-values until they converge below a threshold (`θ = 1e-9`).
- At each sweep, update each state's value by selecting the stake (action) that maximizes expected return.
- Extract the policy corresponding to the optimal value function.

## Output

The program produces and saves a plot (`figure_4_3.png`) showing:

- **Top Plot**: Value function over successive sweeps of value iteration.
- **Bottom Plot**: Final optimal policy — stake vs capital.

Example visualization is inspired by *Figure 4.3* from Sutton and Barto’s textbook.

## How to Run

Ensure you have the required packages installed:
```bash
pip install matplotlib numpy
```

Then run the script with a Python environment supporting Jupyter-style cells or refactor into a `.py` script.


