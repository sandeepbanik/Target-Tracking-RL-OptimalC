# -*- coding: utf-8 -*-
"""
training pipeline
"""
#%%
import vehicle_env as vehicle
from stable_baselines3 import SAC
import pdb
import config as config
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from mcap_callback import MCAPMetricsCallback
from stable_baselines3.common.logger import configure

#%%
# Create environment
env = vehicle.VehicleEnv()
env = Monitor(env)

# Optional: SB3 native logs for plots (CSV + TensorBoard)
log_dir = "./logs_sac"
logger = configure(log_dir, ["stdout", "csv", "tensorboard"])


#initialise SAC agent
model = SAC("MlpPolicy", env, 
            verbose=1, 
            learning_rate=config.learning_rate, 
            buffer_size=config.buffer_size, 
            batch_size=config.batch_size, 
            tau=0.005, 
            gamma=0.99,
            learning_starts = 500)

model.set_logger(logger)

mcap_cb = MCAPMetricsCallback(
    mcap_path="sac_training_full.mcap",
    metrics_every_n_steps=500,
    transitions_every_n_steps=20,
)

# Train the policy
print("Starting training...")
model.learn(total_timesteps=50000, callback=mcap_cb)


# Save the trained model
model.save("sac_robot_policy")
print("Training complete and model saved.")

