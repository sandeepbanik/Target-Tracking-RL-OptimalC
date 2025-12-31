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
- `control_utils.py`: Optimal control formulation and solver interface :contentReference[oaicite:0]{index=0}
- `vehicle_utils.py`: Vehicle dynamics and SNS kinematics :contentReference[oaicite:1]{index=1}
- `config.py`: Vehicle parameters and cost weights :contentReference[oaicite:2]{index=2}

**Example execution**
- `4WS_SNS_simulate.py`: Solves the optimal control problem and visualizes the resulting trajectory :contentReference[oaicite:3]{index=3}

**Trajectory visualization**
![Optimal control single-agent trajectory](https://github.com/sandeepbanik/Target-Tracking-RL-OptimalC/tree/main/Opt%20singel%20agent/opt_agent.gif)

## Reinforcement Learning (Single Agent, SAC)
A Gymnasium-compatible environment is provided for learning goal-reaching behavior using Soft Actor-Critic. The observation is defined relative to the target, and actions correspond to normalized velocity and steering commands.

**Relevant files**
- `vehicle_env.py`: Gym environment definition :contentReference[oaicite:4]{index=4}
- `training.py`: SAC training pipeline with logging and MCAP support :contentReference[oaicite:5]{index=5}
- `config.py`: RL hyperparameters :contentReference[oaicite:6]{index=6}

**Testing / evaluation**
- Load the trained SAC policy and roll out trajectories within `vehicle_env.py` or custom scripts.

**Trajectory visualization**
- GIF: *[to be added by user]*

## Multi-Agent Reinforcement Learning with CBF
The repository supports multi-agent execution by deploying multiple instances of the learned SAC policy, each controlling a separate vehicle. Safety is enforced using a velocity-level control barrier function that guarantees pairwise collision avoidance while minimally modifying the learned action.

**Key features**
- Shared SAC policy across agents.
- Decentralized CBF filtering based on relative positions.
- Continuous-time safety constraints enforced at each step.

**Relevant files**
- `MA_CBF_test.py`: Multi-agent simulation with CBF filtering and animation export :contentReference[oaicite:7]{index=7}
- `vehicle_env.py`: Shared environment dynamics :contentReference[oaicite:8]{index=8}

**Testing**
- `MA_CBF_test.py` runs a three-agent scenario with distinct initial conditions and targets, logs trajectories, and generates an animation.

**Trajectory visualization**
- GIF: *[to be added by user]*

## Visualization Utilities
All trajectory plots and animations are implemented using Matplotlib, with explicit rendering of the vehicle body and individual wheels.

**Relevant file**
- `plot_utils.py` :contentReference[oaicite:9]{index=9}
