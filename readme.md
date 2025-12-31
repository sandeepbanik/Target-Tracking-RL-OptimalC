# Four-Wheel Steering Vehicle Control and Learning Framework

## Summary
This repository implements a four-wheel steering (4WS) vehicle model and provides a unified framework for optimal control, single-agent reinforcement learning, and multi-agent reinforcement learning with safety guarantees via control barrier functions (CBFs). The codebase supports trajectory planning, policy learning using Soft Actor-Critic (SAC), and safe multi-agent execution with collision avoidance.

## Description
The repository contains:
- A kinematic 4WS vehicle model with symmetric negative steering (SNS).
- Optimal control using nonlinear trajectory optimization - Python Control Systems Library.
- A Gym-compatible environment for reinforcement learning.
- Single-agent SAC training and evaluation.
- Multi-agent deployment using a shared SAC policy augmented with CBF acting on the velocity of the model for collision avoidance.
- Visualization utilities for trajectories and animations.


## Optimal Control
The optimal control module formulates a constrained (target as the final state to be reached) nonlinear optimal control problem for the 4WS vehicle using quadratic state and input costs, with actuator saturation constraints. The implementation utilizes on the Python Control Systems Library.

**Relevant files**
- `control_utils.py`: Optimal control formulation and solver
- `vehicle_utils.py`: Vehicle dynamics and SNS kinematics 
- `config.py`: Vehicle parameters and cost weights

**Example execution**
- `4WS_SNS_simulate.py`: Solves the optimal control problem and visualizes the resulting trajectory

**Trajectory visualization**
![Optimal control trajectory](https://github.com/sandeepbanik/Target-Tracking-RL-OptimalC/blob/main/Opt%20singel%20agent/opt_agent.gif)


## Reinforcement Learning (Single Agent, SAC)
A Gym environment is provided for learning to reach a specified target using Soft Actor-Critic algorithm. The observation is defined relative to the target, and actions correspond to normalized velocity and steering commands.

**Relevant files**
- `vehicle_env.py`: Gym environment definition
- `training.py`: SAC training pipeline with logging and MCAP support
- `config.py`: RL hyperparameters

**Testing / evaluation**
- `testing_single_agent.py`:   Load the trained SAC policy and roll out trajectories

**Trajectory visualization**
![Single agent RL](https://github.com/sandeepbanik/Target-Tracking-RL-OptimalC/blob/main/RL%20singel%20and%20multi-agent/single_agent.gif)

## Multi-Agent Reinforcement Learning with CBF
The code also supports multi-agent execution by deploying multiple instances of the learned SAC policy, each controlling a separate vehicle. Safety is enforced using a velocity-level control barrier function that ensures pairwise collision avoidance while minimally modifying the learned action under the assumption that each vehicle is in the safe set (to ensure forward invariance).

**Key features**
- Shared SAC policy across agents.
- CBF control filter.
- Continuous-time safety constraints at each step.

**Relevant files**
- `vehicle_env.py`: Shared environment dynamics

**Testing**
- `MA_CBF_test.py` runs a three-agent scenario with distinct initial conditions and targets, logs trajectories, and generates an animation.

**Trajectory visualization**
![Multi-agent RL](https://github.com/sandeepbanik/Target-Tracking-RL-OptimalC/blob/main/RL%20singel%20and%20multi-agent/three_agent.gif)

## Visualization Utilities
All trajectory plots and animations are implemented using Matplotlib, with explicit rendering of the vehicle body and individual wheels.

## Data Logging (MCAP) and Playback

Training metrics, episode statistics, and transition tuples can be logged to an MCAP file during SAC training. The training pipeline in `training.py` uses an MCAP callback to write:

- `/rl/episode`: episode return and episode length statistics
- `/rl/metrics`: periodic training metrics (e.g., actor/critic loss if available)
- `/rl/transition`: sampled transitions (state, action, reward, next_state, done)

**Relevant files**
- `training.py`: writes MCAP logs via `MCAPMetricsCallback` (e.g., `sac_training_full.mcap`) 
- `data_playback.py`: reads the MCAP file and plots training curves and state trajectories 

### Playback (plots from MCAP)
Use `data_playback.py` to load the MCAP log and plot:
- episode return vs steps
- selected training metrics vs steps
- (x, y) trajectory from logged transitions
- selected state components vs steps

**Example**
python data_playback.py --mcap sac_training_full.mcap


### Playback training 
In addition to MCAP-based logging, training curves recorded by Stable-Baselines3 can be visualized directly from the CSV logs generated during training.

**Relevant file**
- `plot_data.py`: reads SB3 `progress.csv` and plots reward and loss curves :contentReference[oaicite:0]{index=0}

The script loads `./logs_sac/progress.csv` and plots, when available:
- mean episode reward (`rollout/ep_rew_mean`)
- critic loss (`train/critic_loss`)
- actor loss (`train/actor_loss`)
- entropy coefficient (`train/ent_coef`)

**Usage**
```bash
python plot_data.py


