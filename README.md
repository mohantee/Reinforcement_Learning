# Reinforcement Learning

This repository contains two main components focusing on Reinforcement Learning concepts: straightforward Policy Evaluation using the Bellman Equation and a complex Deep Q-Network (DQN) implementation for autonomous driving.

## 1. Bellman Equation Matrix Calculation
**File:** `Bellman Equation.ipynb`

This notebook demonstrates **Iterative Policy Evaluation** on a simple 4x4 GridWorld environment.

### Key Features:
- **Environment:** A 4x4 grid (N=4) representing states (0,0) to (3,3).
- **Goal:** Reach state (3,3).
- **Dynamics:** 
  - Rewards: -1 for every non-terminal step.
  - Actions: Up, Down, Left, Right.
- **Algorithm:**
  - Implements the Bellman Expectation Equation update rule.
  - Averages the value over all 4 possible actions (probability 0.25 each).
  - Iteratively updates the Value Function `V(s)` until convergence (`delta < theta`).
- **Visualization:** Uses `matplotlib` to display the final Value Function heatmap and values for each state.

---

## 2. NeuralNav: Autonomous Car Navigation (DQN)
**File:** `citymap_assignment.py`

A comprehensive interactive simulation of a self-driving car learning to navigate a city map using **Deep Q-Networks (DQN)**.

### Overview:
This application uses PyTorch for the RL agent and PyQt6 for the visualization. It is designed as an educational assignment where the goal is to fine-tune hyperparameters to enable successful navigation.

### Architecture:
- **Agent:** Deep Q-Network (`DrivingDQN`) with 3 hidden layers (128, 256, 128 units).
- **State Space (Input Dim: 9):**
  - 7 Ray-cast distance sensors (detecting obstacles/walls).
  - Angle to the current target.
  - Distance to the current target.
- **Action Space (5 actions):** 
  - Turn Left, Go Straight, Turn Right, Sharp Left, Sharp Right.
- **Learning Algorithm:**
  - Experience Replay with Prioritized buffering for high-reward episodes.
  - Double DQN-like Target Network updates.
  - Epsilon-Greedy exploration strategy.

### Assignment Task:
The code contains several intentionally incorrect parameters marked with **"FIX ME"**.
**Objective:** Find these comments and adjust the values (physics, learning rate, discounts, etc.) so the car can successfully learn to reach the targets without crashing.

### How to Run:
```bash
python citymap_assignment.py
```
*Note: Requires `torch`, `numpy`, and `PyQt6`.*

### Controls:
- **Left Click:** Place the Car and Targets on the map.
- **Right Click:** Finish setup and prepare for training.
- **Spacebar:** Start/Pause training loop.
- **Reset Button:** Clear the map and simulation state.
