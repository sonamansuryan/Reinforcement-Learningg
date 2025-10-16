# NPUA Reinforcement Learning Course Projects

## Overview  
This repository includes a collection of projects developed during the Reinforcement Learning (RL) course at the National Polytechnic University of Armenia (NPUA). The projects integrate theoretical knowledge with practical applications, aiming to develop a deep understanding of decision-making systems. The work is based on the latest methodologies in reinforcement learning research.

---

### What is Reinforcement Learning?

Reinforcement Learning (RL) studies how agents learn to make decisions by interacting with an environment. The agent chooses actions, observes the outcomes, and adapts its strategy to maximize cumulative reward. Unlike traditional supervised learning, RL relies on trial and error rather than labeled data. It is widely applied in robotics, games, and autonomous systems where sequential decision-making is crucial.

---
## Projects


### [Project 1: Tic-Tac-Toe](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/tic-tac-toe)

**Description:** Demonstrates how an agent learns to play optimally via self-play and tabular value updates, showcasing core RL concepts like value estimation, policy improvement, and strategic convergence.

---

### [Project 2: 10-armed-testbed](https://github.com/ZhasminHovhannisyan/Reinforcement-Learning/tree/main/ten-armed-testbed)
**Description:** Simulates a multi-armed bandit problem to compare different action-selection strategies such as ε-greedy, UCB, and gradient methods. The project visualizes average rewards and optimal action rates over time to build intuition around exploration-exploitation dynamics.

---

### [Project 3: Gridworld - MDP](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/gridworld-mdp)
**Description:** Implements a simple grid-based Markov Decision Process (MDP) to visualize value functions and policies. Includes simulations under both random and optimal policies using iterative evaluation and value iteration. Ideal for understanding how rewards and transitions shape decision-making in finite environments.

---

### [Project 4: Gridworld - Dynamic Programming](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/gridworld-dpp)
**Description:** Solves a 4x4 gridworld task using policy evaluation, improvement, and iteration. Demonstrates how an agent can learn the optimal policy by computing value functions iteratively. Great for understanding dynamic programming methods in episodic, undiscounted environments.

---

### [Project 5: Gambler’s Problem – Value Iteration](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/gambler-problem)

**Description:** Solves an episodic betting task using value iteration to find the optimal policy. Demonstrates how a gambler can maximize returns by betting strategically in a biased coin-flip game. Highlights stochasticity, policy convergence, and the balance between risk and reward.

---

### [Project 6: Blackjack - Monte Carlo Methods](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/blackjack)

**Description:** Models Blackjack as an episodic MDP and applies Monte Carlo techniques for policy evaluation, improvement, and off-policy learning. Demonstrates learning optimal strategies through Exploring Starts and evaluates sampling efficiency with ordinary vs. weighted importance sampling.

---

### [Project 7: Infinite - variance](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/infinite-variance)

**Description:**  Illustrates instability in ordinary importance sampling due to infinite variance in off-policy Monte Carlo estimation. Demonstrates how weighted importance sampling provides more stable convergence when estimating value functions under stochastic looping behavior.

---

### [Project 8: Random Walk — TD(0) vs Monte Carlo](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/random-walk)

**Description:** This project compares TD(0) and Monte Carlo prediction methods in a simple Markov Reward Process. It highlights the trade-offs between bootstrapping (TD) and full-return updates (MC) in estimating state values from experience.

---

### [Project 9: Windy Gridworld — SARSA](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/windy-gridworld)

**Description:** This project uses the on-policy TD control algorithm SARSA to solve the Windy Gridworld environment. The agent learns to reach a goal efficiently despite wind disturbances. It demonstrates how exploration and temporal-difference learning can produce near-optimal paths in dynamic, stochastic environments.

---

### [Project 10: Cliff-walking](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/cliff-walking)

**Description:** This project compares SARSA (on-policy) and Q-learning (off-policy) in the Cliff Walking gridworld. It shows how SARSA learns a safer path by accounting for exploration, while Q-learning aims for optimality but risks falling into the cliff. The experiment demonstrates the trade-off between safety during learning and final policy performance under ε-greedy exploration.  

---

### [Project 11: Maximization Bias](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/maximization-bias)

**Description:**
Explores the phenomenon of **maximization bias** in reinforcement learning. Demonstrates how estimating the maximum expected value from samples can introduce a positive bias, and how Double Q-learning mitigates this issue. The project highlights the difference between **overestimation** and **unbiased value estimation** in action-value methods.

---

### [Project 12: Random Walk — N-step TD](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/random-walk-ntd)

**Description:**
Implements **n-step TD prediction** on the 1000-state random walk example. The project compares TD(0), Monte Carlo, and intermediate n-step methods, showing how bootstrapping multiple steps improves learning speed and reduces variance. Visualization includes learning curves over different n-step sizes.

---

### [Project 13: Mazes — Planning and Learning](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/mazes)

**Description:**
Demonstrates **model-based RL** using simple maze environments. Implements **Dyna-Q**, prioritized sweeping, and basic planning algorithms. Highlights how simulated experience (planning) combined with real experience accelerates learning optimal paths in stochastic environments.

---

### [Project 14: Updates Comparison — Monte Carlo vs TD](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/updates-comparison)

**Description:**
Compares **Monte Carlo** and **temporal-difference (TD) updates** in a controlled setting. The project illustrates differences in bias, variance, and learning speed between the two approaches, using simple Markov Reward Processes for demonstration.

---

### [Project 15: Trajectory Sampling](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/trajectory-sampling)

**Description:**
Implements **trajectory sampling** methods to approximate value functions in episodic tasks. Demonstrates how sampled trajectories can be used for policy evaluation, and explores the trade-offs between sample efficiency and variance in predictions.

---

### [Project 16: Random Walk — Function Approximation](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/random-walk-fa)

**Description:**
Extends the 1000-state random walk example using **linear function approximation**. Implements polynomial and Fourier bases to represent value functions. Highlights challenges in generalization and feature design, and compares learning curves for different basis functions.

---

### [Project 17: Coarse Coding](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/coarse-coding)

**Description:**
Demonstrates **coarse coding** (tile coding) for linear function approximation on a 1D square-wave function. Explores the effect of **feature width** on generalization and learning. Wider features produce smooth generalization, whereas narrow features produce localized, bumpy estimates. The project highlights the role of receptive field design in function approximation.

---

##  Reference

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).  
  🔗 [Read the full book (PDF)](http://incompleteideas.net/book/RLbook2020.pdf)

---

##  Educational Objective

This repository is intended as a companion to theoretical learning, providing hands-on experience with classical reinforcement learning algorithms and concepts. Each project serves as a stepping stone toward mastering more advanced topics in AI and autonomous decision-making.




