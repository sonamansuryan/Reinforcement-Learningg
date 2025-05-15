# NPUA Reinforcement Learning Course Projects

## Overview  
This repository includes a collection of projects developed during the Reinforcement Learning (RL) course at the National Polytechnic University of Armenia (NPUA). The projects integrate theoretical knowledge with practical applications, aiming to develop a deep understanding of decision-making systems. The work is based on the latest methodologies in reinforcement learning research.

---

## Projects

---

### [Project 1: Tic-Tac-Toe](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/tic-tac-toe)

**Description:** Demonstrates how an agent learns to play optimally via self-play and tabular value updates, showcasing core RL concepts like value estimation, policy improvement, and strategic convergence.

**Based on:** [Section 1.1 – Tic-Tac-Toe](http://incompleteideas.net/book/RLbook2020.pdf#page=41)

---

### [Project 2: 10-armed-testbed](https://github.com/ZhasminHovhannisyan/Reinforcement-Learning/tree/main/ten-armed-testbed)
**Description:** Simulates a multi-armed bandit problem to compare different action-selection strategies such as ε-greedy, UCB, and gradient methods. The project visualizes average rewards and optimal action rates over time to build intuition around exploration-exploitation dynamics.

**Based on:** [Example 2.2 – Ten-Armed Testbed](http://incompleteideas.net/book/RLbook2020.pdf#page=65)

---

### [Project 3: Gridworld - MDP](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/gridworld-mdp)
**Description:** Implements a simple grid-based Markov Decision Process (MDP) to visualize value functions and policies. Includes simulations under both random and optimal policies using iterative evaluation and value iteration. Ideal for understanding how rewards and transitions shape decision-making in finite environments.

**Based on:** [Chapter 3 – Finite Markov Decision Processes](http://incompleteideas.net/book/RLbook2020.pdf#page=91)

---

### [Project 4: Gridworld - Dynamic Programming](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/gridworld-dpp)
**Description:** Solves a 4x4 gridworld task using policy evaluation, improvement, and iteration. Demonstrates how an agent can learn the optimal policy by computing value functions iteratively. Great for understanding dynamic programming methods in episodic, undiscounted environments.

**Based on:** [Example 4.1 – Gridworld](http://incompleteideas.net/book/RLbook2020.pdf#page=113)

---

### [Project 5: Gambler’s Problem – Value Iteration](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/gambler-problem)

**Description:** Solves an episodic betting task using value iteration to find the optimal policy. Demonstrates how a gambler can maximize returns by betting strategically in a biased coin-flip game. Highlights stochasticity, policy convergence, and the balance between risk and reward.

**Based on:** [Example 4.3 – Gambler’s Problem](http://incompleteideas.net/book/RLbook2020.pdf#page=118)

---

### [Project 6: Blackjack - Monte Carlo Methods](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/blackjack)

**Description:** Models Blackjack as an episodic MDP and applies Monte Carlo techniques for policy evaluation, improvement, and off-policy learning. Demonstrates learning optimal strategies through Exploring Starts and evaluates sampling efficiency with ordinary vs. weighted importance sampling.

**Based on:** [Example 5.1 – Blackjack](http://incompleteideas.net/book/RLbook2020.pdf#page=134)

---

### [Project 7: Infinite - variance](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/infinite-variance)

**Description:**  Illustrates instability in ordinary importance sampling due to infinite variance in off-policy Monte Carlo estimation. Demonstrates how weighted importance sampling provides more stable convergence when estimating value functions under stochastic looping behavior.

**Based on:** [Section 5.4 – Off-policy Prediction](http://incompleteideas.net/book/RLbook2020.pdf#page=147)

---

### [Project 8: Random Walk — TD(0) vs Monte Carlo](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/random-walk)

**Description:** This project compares TD(0) and Monte Carlo prediction methods in a simple Markov Reward Process. It highlights the trade-offs between bootstrapping (TD) and full-return updates (MC) in estimating state values from experience.

**Based on:** [Example 6.2 – Random Walk](http://incompleteideas.net/book/RLbook2020.pdf#page=140)

---

### [Project 9: Windy Gridworld — SARSA](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/windy-gridworld)

**Description:** This project uses the on-policy TD control algorithm SARSA to solve the Windy Gridworld environment. The agent learns to reach a goal efficiently despite wind disturbances. It demonstrates how exploration and temporal-difference learning can produce near-optimal paths in dynamic, stochastic environments.

**Based on:** [Example 6.5 – Windy Gridworld](http://incompleteideas.net/book/RLbook2020.pdf#page=132)

---

### [Project 10: Cliff-walking](https://github.com/sonamansuryan/Reinforcement-Learningg/tree/main/cliff-walking)

**Description:** This project compares SARSA (on-policy) and Q-learning (off-policy) in the Cliff Walking gridworld. It shows how SARSA learns a safer path by accounting for exploration, while Q-learning aims for optimality but risks falling into the cliff. The experiment demonstrates the trade-off between safety during learning and final policy performance under ε-greedy exploration.  

**Based on:** [Example 6.6 – Cliff Walking](http://incompleteideas.net/book/RLbook2020.pdf#page=145)

---

##  Reference

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).  
  🔗 [Read the full book (PDF)](http://incompleteideas.net/book/RLbook2020.pdf)

---

##  Educational Objective

This repository is intended as a companion to theoretical learning, providing hands-on experience with classical reinforcement learning algorithms and concepts. Each project serves as a stepping stone toward mastering more advanced topics in AI and autonomous decision-making.

