
# Gridworld: Policies, Value Functions, and Optimality (Figures 3.2 & 3.5)

This project implements and visualizes concepts from **Chapter 3** of *Reinforcement Learning: An Introduction* by Sutton & Barto. It focuses on evaluating the **state-value function** under a fixed policy and computing **optimal value functions** and **policies** via dynamic programming.

---

## Environment Overview

- A rectangular **5x5 gridworld** representing a simple Markov Decision Process (MDP).
- At each grid cell (state), the agent can take **4 actions**: north, south, east, and west.
- **Deterministic transitions**: moving in the direction of the action unless it leads off the grid.
  - Off-grid actions result in:
    - No movement
    - A reward of **-1**
  - All other movements yield:
    - **0 reward**
- **Special states**:
  - **A**: All actions yield **+10** and move the agent to **A′**
  - **B**: All actions yield **+5** and move the agent to **B′**

---

## Tasks

### 1. **Policy Evaluation (Figure 3.2)**

- Agent follows a **uniform random policy** (equiprobable actions in all states).
- **Discount factor** γ = 0.9.
- Value function `v_π(s)` is computed iteratively using the Bellman expectation equation.
- Converged value function reflects expected return under the random policy.

> Notable insight: negative values in lower states indicate the risk of hitting the edge frequently under the random policy.

### 2. **Optimal Value Function and Policy (Figure 3.5)**

- Solve the **Bellman optimality equation** to compute the **optimal state-value function** `v_*`.
- From `v_*`, derive the **optimal policy** by selecting actions that maximize expected return from each state.
- Visualization:
  - **Optimal value function** grid
  - **Optimal policy** with arrows indicating best actions (can be multiple per state)

---

##  Outputs

- `figure_3_2.png`: Value function under random policy.
- `figure_3_5.png`: Optimal state-value function.
- `figure_3_5_policy.png`: Optimal policy visualization (arrows per cell).

---

##  How to Run

1. Ensure Python dependencies are installed:
```bash
pip install numpy matplotlib
```

2. Place the file `grid_world.py` in the `src/` folder with implementations for:
   - `grid_size`: Gridworld dimensions
   - `actions`: List of action directions
   - `step(state, action)`: Transition function returning `(next_state, reward)`
   - `draw(grid, is_policy=False)`: Renders the grid (with value or policy arrows)

3. Execute the notebook or script to generate the figures.

---

##  References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)  
  [Download the book (free)](http://incompleteideas.net/book/RLbook2020.pdf)



