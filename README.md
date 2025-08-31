# Reinforcement Learning for Satellite Collision Avoidance using PPO

## Overview
This project explores the use of **Proximal Policy Optimization (PPO)** for collision avoidance maneuvers (CAM) in Low Earth Orbit (LEO) while minimizing both **delta-V** and orbital deviations. The environment is highly simplified, representing a **2D Keplerian orbit without perturbations** and a single debris object. The PPO implementation is minimal, serving as a baseline for reinforcement learning in this context.

## Installation
Clone the repository and install the dependencies:
```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

Ensure you are using **Python 3.12.3** (or a compatible version).

## Project Structure

- **Environment.py**  
  Defines the environment simulating a 2D Keplerian LEO orbit and a debris object. Running this script will plot an example scenario, including the nominal satellite orbit and debris trajectory with collision at TCA.

- **ActorCritic.py**  
  Implements the actor-critic neural network used by PPO.

- **PPO.py**  
  Implements the PPO-clip algorithm, including advantage normalization and entropy regularization.

- **utils.py**  
  Utility functions for plotting results and other helper methods.

- **train.py**  
  Implements the main training loop for the baseline hyperparameters specified in `config.yaml`.  
  Outputs:
  - Logs in `logs/` (CSV format)
  - Models in `models/` (`.pth` files)

- **main.py**  
  Runs the training loop for sensitivity analysis. The hyperparameter to tune and its values must be specified manually at the beginning of the script. Results are saved in subfolders under `logs/` and `models/` named after the hyperparameter.

- **test.py**  
  Used to evaluate a trained model and visualize its performance.  
  **Capabilities:**  
  - Plot (on the same axis) average rewards vs episode from log files
  - Compute **average reward, delta-V, and episode duration** over multiple episodes.
  - Plot CAM trajectory for a selected model.
  - Plot **orbital element divergence** over the episode.
  - Optionally **show or save the plots**.

  **Example usage:**
  ```bash
  # Plot average episodic reward resulting from sensitivity analysis on discount factor
  python3 test.py --model clip_ratio --action

  # Plot CAM trajectory for a given model
  python3 test.py --model baseline --action test --num_episodes 1

  # Compute average stats for 100 episodes
  python3 test.py --model baseline --action test --num_episodes 100

  # Plot orbital element divergence
  python3 test.py --model baseline --action test --episodes 1 --plot_divergence
  ```
  **Arguments:**
  - `--model` : Relative path to `models/` (`logs/`) of the trained model file (log file) if `--action` is `test` (`plot`)
  - `--action` : Whether to plot log files or test a model (default: plot)
  - `--num_episodes` : Number of test episodes
  - `--show` : Display plots (default: True)
  - `--save` : Save plots to file (default: True)
  - `--plot_divergence` : Plot divergence of orbital elements over time (default: False)

## Configuration
- All baseline hyperparameters are stored in `config.yaml`.
- Modify this file to change PPO settings.
