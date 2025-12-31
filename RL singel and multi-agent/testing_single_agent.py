# -*- coding: utf-8 -*-
"""
testing environment
"""
#%%
from stable_baselines3 import SAC
import vehicle_env as vehicle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from helper import rot2
import config
import plot_utils as pu
#%%

# Load the environment and the trained model
env = vehicle.VehicleEnv()
model = SAC.load("sac_robot_policy")

# Reset the environment.
obs, _ = env.reset()
states = []
controls = []
time_span = []
done = False

print("Simulating test episode...")
for _ in range(config.max_steps): # Max steps for the test
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    states.append(env.state.copy()) # Store the state
    controls.append(action.copy()) # Store the control
    time_span.append(env.current_time) # Store the time
    
    if terminated or truncated:
        break

#%% Plotting

# Plotting trajectory.
target = env.target
states_pt = np.array(states).T
controls_pt = np.array(controls).T
pu.plot_vehicle_to_target(env, states_pt, controls_pt, target, time_span)

