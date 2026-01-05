from stable_baselines3 import SAC
import vehicle_env as vehicle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from helper import rot2
import config
import plot_utils as pu
from mcap_logger import McapLogger, sec_to_ns
import time as time
import mcap_utils as mc

import math


env = vehicle.VehicleEnv() # Create environment
model = SAC.load("sac_robot_policy") # Load trained policy

obs, _ = env.reset() # Reset environment
states = [] # Store the state
controls = [] # Store the control inputs
time_span = [] # Store the time stamps

# Create logger and register topics
logger = McapLogger("singel_agent_episode.mcap") # Save file as test_episode.mcap
# Register schemas and channels
state_schema_id = logger.register_schema("RobotState", mc.STATE_SCHEMA)
ctrl_schema_id = logger.register_schema("RobotControl", mc.ACT_SCHEMA)
logger.register_channel("/robot/state", state_schema_id)
logger.register_channel("/robot/control", ctrl_schema_id)

# Logger scene update.
scene_schema_id = logger.register_schema("foxglove.SceneUpdate", mc.SCENEUPDATE_SCHEMA_MIN)
logger.register_channel("/robot/scene", scene_schema_id)


ep_schema_id = logger.register_schema("Episode", mc.EPISODE_SCHEMA) # Register episode schema
logger.register_channel("/robot/episode", ep_schema_id) # Register episode channel

t0_wall_ns = sec_to_ns(time.time()) # Wall clock time at start
t0_sim = float(env.current_time) # Simulation time at start

episode_id = 0
target = np.asarray(env.target).ravel()  
logger.write("/robot/episode", t0_wall_ns, {
    "episode_id": episode_id,
    "target": [float(v) for v in target],
    "t0": float(env.current_time),
})

# Simulate an episode.
for _ in range(config.max_steps):
    action, _ = model.predict(obs, deterministic=True) # Get action from policy
    obs, reward, terminated, truncated, info = env.step(action) # Step environment

    # store for your existing plotting
    states.append(env.state.copy())
    controls.append(action.copy())
    time_span.append(env.current_time)

    # write to MCAP
    t_sec = float(env.current_time)
    t_ns = t0_wall_ns + sec_to_ns(t_sec - t0_sim)

    s = np.asarray(env.state).ravel()
    if s.size < 3:
        raise RuntimeError(f"env.state must contain at least [x,y,yaw], got size={s.size}")
    
    msg_state = {
        "episode_id": 0,
        "t": t_sec,
        "x": float(s[0]),
        "y": float(s[1]),
        "yaw": float(s[2]),
        "v": float(s[3]) if s.size > 3 else 0.0,
        "state": [float(v) for v in s],
    }
    logger.write("/robot/state", t_ns, msg_state)

    msg_u = {
        "episode_id":0,
        "t": t_sec,
        "u_sac": [float(v) for v in np.asarray(action).ravel()]
    }

    # Wheel data
    w_d = env.wheel_delta.copy()
    
    msg_u["delta_fl"] = float(w_d[0])
    msg_u["delta_fr"] = float(w_d[1])
    msg_u["delta_rl"] = float(w_d[2])
    msg_u["delta_rr"] = float(w_d[3])

    logger.write("/robot/control", t_ns, msg_u)

    scene_msg = mc.make_vehicle_scene_update(
    x=msg_state["x"],
    y=msg_state["y"],
    yaw=msg_state["yaw"],
    wheel_delta=w_d,
    target=np.asarray(env.target).ravel(),
    L=float(config.L),
    W=float(config.W),
    )
    logger.write("/robot/scene", t_ns, scene_msg)

    if terminated or truncated:
        break

logger.close()

# Your existing arrays still work
states_pt = np.array(states).T
controls_pt = np.array(controls).T