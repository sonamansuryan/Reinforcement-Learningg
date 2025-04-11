# Gridworld: Policy Evaluation and Policy Improvement

This project demonstrates **Iterative Policy Evaluation**, **Policy Improvement**, and **Policy Iteration** using the classic **4x4 Gridworld** example from Chapter 4 of *Reinforcement Learning: An Introduction* by Sutton and Barto.

## Environment Description

- A **4x4 gridworld** with **16 states**:
  - **Terminal states**: top-left and bottom-right corners (states 0 and 15).
  - **Non-terminal states**: numbered from 1 to 14.
- **Actions**: Up, Down, Left, Right — deterministic.
  - Actions that would take the agent off the grid leave the agent in the same state.
- **Reward**: -1 for every action until a terminal state is reached.
- **Policy**: Equiprobable random policy — each action is chosen with equal probability.

## Tasks

### Policy Evaluation
- Estimate the **state-value function** `v_π(s)` for the given random policy using **iterative updates**.
- Two versions implemented:
  - **In-place** updates: updates are immediately used in subsequent state calculations.
  - **Out-of-place** updates: all updates are stored in a copy and applied simultaneously after each sweep.
- Convergence is checked by comparing the maximum difference between value estimates across iterations.

### Policy Improvement
- Once `v_π` is estimated, derive a **greedy policy** `π'` that maximizes value with respect to `v_π`.
- The result illustrates **policy improvement**, showing that `π'` performs better or equally well compared to `π`.

###  Policy Iteration (Discussion)
- In this specific Gridworld, applying policy improvement on the random policy results in an **optimal policy** in just one iteration.

## Outputs

- `figure_4_1_in_place.png`: Value function obtained using in-place updates.
- `figure_4_1_out_place.png`: Value function from out-of-place updates.
- Both visualizations show rounded value estimates for each grid cell.

## How to Run

1. Ensure dependencies are installed:
```bash
pip install numpy matplotlib
```

2. Place the `grid_world.py` file inside a `src/` directory. It should implement:
   - `compute_state_value(in_place=True/False)` – returns value function and number of iterations.
   - `draw(grid)` – visualizes the value function over the 4x4 grid.

3. Run the main script in a Jupyter environment or refactor to `.py` as needed.

## Reference

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.) – [Free PDF](http://incompleteideas.net/book/RLbook2020.pdf)

---

