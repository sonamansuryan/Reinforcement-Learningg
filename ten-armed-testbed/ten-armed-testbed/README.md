# Multi-Armed Bandit Reinforcement Learning Project

## Overview
This project explores the Multi-Armed Bandit problem, a fundamental concept in Reinforcement Learning, using epsilon-greedy strategies. The implementation compares different epsilon values and examines the tradeoff between exploration and exploitation. Additionally, the project analyzes the effects of optimistic initial values in stationary environments.

## Key Components
- **Bandit Class**: Implements the multi-armed bandit problem with various strategies.
- **Epsilon-Greedy Strategies**: Compares different epsilon values (ε = 0.00, ε = 0.01, ε = 0.10) to analyze their effects on learning performance.
- **Optimistic Initialization**: Evaluates the impact of setting initial Q-values optimistically (e.g., Q₁(a) = 5 for ε = 0).
- **Performance Analysis**: Measures reward efficiency in stationary environments, highlighting how optimistic initialization encourages early exploration.
- **Simulation and Visualization**: Runs experiments and generates plots to compare different approaches.

## Operating Modes
### 1. Exploration vs. Exploitation
- The agent balances discovering new strategies (exploration) and leveraging known best actions (exploitation).

### 2. Stationary Environment Testing
- Evaluates strategies in a stable environment where the reward distributions do not change over time.

## Dependencies
- `numpy` - For numerical computations
- `matplotlib` - For plotting results
- `tqdm` - For progress tracking

## Conclusion
This project demonstrates how different reinforcement learning strategies perform in a multi-armed bandit scenario. The results highlight that optimistic initialization encourages early exploration, making it an effective approach in stationary environments.

