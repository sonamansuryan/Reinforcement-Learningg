# Tic-Tac-Toe Reinforcement Learning Project


##  Overview

This project is a practical implementation of Reinforcement Learning applied to the classic Tic-Tac-Toe game. The system builds an intelligent agent that learns optimal gameplay through experience and policy refinement.

The AI learns from self-play and stores optimal decisions in serialized policy files for future use.

---

##  Key Components

- **Learning Agent (`player.py`)**  
  Implements reinforcement learning (value-based) to improve strategies over time. Learns via self-play and stores policies in `.bin` files.
  
- **Game Logic & State Tracker (`state.py`)**  
  Manages board configuration, transitions, and state encoding.
  
- **Judge (`judge.py`)**  
  Checks for win/draw conditions and evaluates game outcomes.
  
- **Execution Environment (`tic_tac_toe.py`)**  
  The main script to run training, play, or evaluation. Acts as a controller.

- **Policies**  
  Pretrained policies (`policy_first.bin`, `policy_second.bin`) store learned value functions for both players.

---

##  Operating Modes

### ➤ Training Mode
- The AI trains via repeated self-play.
- Value functions are updated using rewards from game outcomes.
- Policies are saved for later inference/play.

### ➤ Play Mode
- A human can play against a trained agent.
- Alternatively, two AI agents can play against each other.
- Real-time decision-making is done using stored policy files.

---

## How to Use

### Requirements

- Python 3.10 +
- No external dependencies required (only standard library)

### ▶ Run the Game

```bash
cd tic-tac-toe/src
python tic_tac_toe.py
````

Follow the on-screen instructions to choose training or play mode.

---

##  File Structure

```
tic-tac-toe/
├── src/
│   ├── tic_tac_toe.py         # Main script
│   ├── state.py               # Game state logic
│   ├── player.py              # AI and human players
│   ├── judge.py               # Outcome evaluation
│   ├── policy_first.bin       # Policy for player 1 (AI)
│   ├── policy_second.bin      # Policy for player 2 (AI)
```

---

## Notes

* The agent uses a table-based value function approach, storing the estimated value for each state.
* Policy files are stored as binary data using Python's `pickle` module.
* Retraining will overwrite existing policies unless backed up.
