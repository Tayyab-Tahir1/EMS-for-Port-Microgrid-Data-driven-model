# Energy Management System Reinforcement Learning

This repository contains code for training and evaluating reinforcement learning agents for an Energy Management System (EMS) application. The EMS simulates the operation of an energy system with components such as a battery storage system, hydrogen supply, and photovoltaic (PV) generation, using dynamic pricing (e.g., Time-of-Use tariffs, Feed-in Tariff) to optimize costs and energy flows.

## Project Overview

The project includes:

- **Custom Environment:**  
  Implemented in `Models/environment.py`, the `EnergyEnv` class simulates energy flows, battery and hydrogen operations, and computes rewards based on cost optimization.

- **Training Scripts:**  
  - `train/trainer.py`: Contains training routines for both Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents.
  - `main.py`: Serves as the entry point for training using either DQN or PPO, with GPU configuration and wandb logging.
  - `train_and_test_dqn_stable.py`: Uses the stable-baselines3 implementation of DQN with a Gymnasium wrapper for training and testing.

- **Validation Scripts:**  
  - `MPC_validate.py`: Uses Model Predictive Control (MPC) with CVXPY as a benchmark controller.
  - `validation_stable.py`: Validates a pre-trained stable-baselines3 DQN model on the EMS environment and generates performance plots.

## Requirements

- Python 3.7+
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://gymnasium.farama.org/)
- [CVXPY](https://www.cvxpy.org/)
- [Weights & Biases (wandb)](https://wandb.ai/)
- Additional dependencies: NumPy, pandas, matplotlib, tqdm, argparse

# To run 
python train_and_test_dqn_stable.py --mode both --timesteps 100000 --test_episodes 10

Install the required packages (you may create a `requirements.txt` based on these dependencies):

```bash
pip install -r requirements.txt
