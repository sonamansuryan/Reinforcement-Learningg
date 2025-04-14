# Blackjack Simulation with Monte Carlo Methods

This repository contains a series of simulations and experiments related to the popular casino card game of Blackjack, using **Monte Carlo Methods (MCMs)** for solving various problems in reinforcement learning. The objective is to evaluate state-value functions, implement policies, and demonstrate off-policy learning using Monte Carlo methods.

## Game Overview

Blackjack is a casino card game in which the objective is to get a hand as close as possible to a total value of 21, without exceeding it. Face cards (Jack, Queen, King) are worth 10 points, and an Ace can either count as 1 or 11. 

In this simulation:
- **The player** competes independently against the dealer.
- **The dealer** follows a fixed strategy: hits if the sum of their cards is below 17 and sticks otherwise.

### Key Concepts
- **Episode**: A complete game of Blackjack.
- **Rewards**: +1 for a win, -1 for a loss, and 0 for a draw.
- **State**: Defined by the player’s total sum, the dealer's face-up card, and whether the player has a usable Ace.

## Projects and Approaches

### 1. **Monte Carlo Prediction**
- The objective was to compute the state-value function for a policy that sticks on sums of 20 or 21 and hits otherwise.
- The state-value function was approximated by running **Monte Carlo simulations** over **10,000** and **500,000** episodes.
- The results were visualized with heatmaps, showing state-value estimates for different combinations of player sum and dealer’s face-up card.

### 2. **Monte Carlo Control**
- We implemented **Monte Carlo Exploring Starts (MC ES)** to explore the optimal policy for Blackjack.
- The optimal policy was derived by simulating random episodes and updating the state-action values.
- The results were compared with the “basic” strategy of Thorp (1966), showing that the derived optimal policy is similar, with a minor discrepancy related to the usable Ace case.

### 3. **Off-policy Prediction via Importance Sampling**
- **Ordinary Importance Sampling** and **Weighted Importance Sampling** were applied to estimate the value of a specific Blackjack state from off-policy data.
- By using random policies to generate data and target policies to evaluate the value of a state, the goal was to compare the effectiveness of these off-policy learning methods.
- The error rates of both methods were visualized, showing that weighted importance sampling produced lower error estimates early in the learning process.

## Files and Functions

### Files:
- `blackjack-card-values.png`: A reference image for Blackjack card values.
- `Figure_5_1.PNG`: Heatmaps of the state-value functions computed for various Blackjack policies.
- `Figure_5_2.PNG`: The optimal policy for Blackjack derived by MC ES.
- `Figure_5_3.PNG`: Learning curves for ordinary and weighted importance sampling.

### Functions:
- `monte_carlo_on_policy(episodes)`: Runs Monte Carlo simulations to evaluate the state-value function for a policy.
- `monte_carlo_es(episodes)`: Implements Monte Carlo Exploring Starts to derive the optimal policy and state-action values.
- `monte_carlo_off_policy(episodes)`: Uses ordinary and weighted importance sampling to estimate the value of a Blackjack state from off-policy episodes.

