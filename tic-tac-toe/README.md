# Tic-Tac-Toe – Reinforcement Learning Project

This repository contains a reinforcement learning implementation of the classic **Tic-Tac-Toe** game. Inspired by early RL research and educational demonstrations, it explores how an agent can **learn optimal gameplay through self-play and value-based methods**.

---

## Background

**Tic-Tac-Toe** is a simple turn-based zero-sum game that offers a small but rich environment to understand **value estimation**, **policy learning**, and **self-play training**. Despite its simplicity, it allows us to explore key reinforcement learning ideas, such as:

* Learning from reward signals
* State value approximation
* Policy improvement over time
* Strategy convergence through experience

This project demonstrates a **tabular value function approach**, where state values are stored explicitly and updated iteratively based on game outcomes.

---

## Structure

* `player.py`: Implements both AI and human agents. The AI uses a reinforcement learning approach to update state values through self-play and stores learned policies in `.bin` files.
* `state.py`: Handles the board configuration, game state representation, and state hashing for policy lookups.
* `judge.py`: Evaluates game termination conditions—win, lose, or draw.
* `tic_tac_toe.py`: Main script that runs training or gameplay sessions. Entry point of the project.
* `policy_first.bin` and `policy_second.bin`: Serialized policy files (value tables) for Player 1 and Player 2 respectively.

---

## Simulation Overview

The learning agent plays thousands of games against itself, gradually improving its strategy:

* **State values** are initialized arbitrarily (often zero) and are updated based on reward signals.
* **Self-play** allows exploration of many board configurations.
* After training, the AI uses the learned value tables (policies) to play optimally or near-optimally.
* Policies are saved to `.bin` files for reuse during play.

Two operating modes:

* **Training Mode**: Run thousands of self-play games to learn value functions.
* **Play Mode**: A human can challenge the trained AI, or two AI agents can compete.

---

## Reference

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd Edition)*.
  * [Chapter 1 – Introduction](http://incompleteideas.net/book/RLbook2020.pdf#page=15)

---

## Educational Objective

This project is designed for **educational purposes** and aims to teach:

* How value-based reinforcement learning works in discrete environments
* The importance of policy evaluation and improvement
* The effectiveness of self-play in learning strategic behavior

By training an agent to master Tic-Tac-Toe, learners gain hands-on experience with RL fundamentals in a controlled, visualizable environment.
