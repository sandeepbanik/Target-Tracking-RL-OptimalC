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
import control_utils as cu
from mcap_logger import McapLogger, sec_to_ns
import mcap_utils as mu


# ---------------- Create 3 agents (each with its own env + same learned SAC policy) ----------------
env_agent1 = vehicle.VehicleEnv()
model_agent1 = SAC.load("sac_robot_policy")

env_agent2 = vehicle.VehicleEnv()
model_agent2 = SAC.load("sac_robot_policy")

env_agent3 = vehicle.VehicleEnv()
model_agent3 = SAC.load("sac_robot_policy")


# ---------------- Reset and optionally overwrite initial conditions for multi-agent setup ----------------
obs1, _ = env_agent1.reset()
obs2, _ = env_agent2.reset()
obs3, _ = env_agent3.reset()

# Optional: enforce your own start/goal layout (recommended for reproducible 3-agent tests)
# Example: same target, different initial poses.
env_agent1.target = np.array([10.0, 10.0], dtype=float)
env_agent2.target = np.array([2.0, 8.0], dtype=float)
env_agent3.target = np.array([4.0, -4.0], dtype=float)

env_agent1.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
env_agent2.state = np.array([5.0, 0.0, 0.0, 0.0], dtype=float)
env_agent3.state = np.array([-5.0, 4.0, 0.0, 0.0], dtype=float)
obs1 = env_agent1._get_obs(); obs2 = env_agent2._get_obs(); obs3 = env_agent3._get_obs()


# CBF parameters
d_min = 2.0
kappa = 1.0

# Create logger and register topics
logger = McapLogger("three_agents_episode.mcap")
# Register schemas
scene_schema_id = logger.register_schema("foxglove.SceneUpdate", mu.SCENEUPDATE_SCHEMA_MIN)
state_schema_id = logger.register_schema("RobotState", mu.STATE_SCHEMA)
ctrl_schema_id = logger.register_schema("RobotControl", mu.ACT_SCHEMA)

# Register channels
logger.register_channel("/scene", scene_schema_id)
for i in [1, 2, 3]:
    logger.register_channel(f"/agent{i}/state", state_schema_id)
    logger.register_channel(f"/agent{i}/control", ctrl_schema_id)



# Use agent1 as the clock reference (they step together)
t0_wall_ns = sec_to_ns(time.time())
t0_sim = float(getattr(env_agent1, "current_time", 0.0))


for step in range(config.max_steps):
    # Snapshot positions
    p1 = np.array([env_agent1.state[0], env_agent1.state[1]], dtype=float)
    p2 = np.array([env_agent2.state[0], env_agent2.state[1]], dtype=float)
    p3 = np.array([env_agent3.state[0], env_agent3.state[1]], dtype=float)

    # Predict
    a1, _ = model_agent1.predict(obs1, deterministic=True)
    a2, _ = model_agent2.predict(obs2, deterministic=True)
    a3, _ = model_agent3.predict(obs3, deterministic=True)

    # CBF
    a1 = cu.cbf_filter_velocity_only_vehicleenv(env_agent1, a1, other_positions_xy=[p2, p3], d_min=d_min, kappa=kappa)
    a2 = cu.cbf_filter_velocity_only_vehicleenv(env_agent2, a2, other_positions_xy=[p1, p3], d_min=d_min, kappa=kappa)
    a3 = cu.cbf_filter_velocity_only_vehicleenv(env_agent3, a3, other_positions_xy=[p1, p2], d_min=d_min, kappa=kappa)

    # Step envs
    obs1, r1, term1, trunc1, info1 = env_agent1.step(a1)
    obs2, r2, term2, trunc2, info2 = env_agent2.step(a2)
    obs3, r3, term3, trunc3, info3 = env_agent3.step(a3)

    # --------- Common timestamp for this step (nanoseconds) ---------
    # Prefer sim time if env provides it; else infer from step index and dt
    if hasattr(env_agent1, "current_time"):
        t_sec = float(env_agent1.current_time)
    else:
        # fallback: if you know dt in config
        t_sec = float(step) * float(config.dt)

    t_ns = t0_wall_ns + sec_to_ns(t_sec - t0_sim)

    # --------- Log /agent{i}/state and /agent{i}/control ---------
    def log_agent(i, env_i, a_i, r_i, info_i):
        s = np.asarray(env_i.state).ravel()
        w_d = np.asarray(env_i.wheel_delta).ravel()
        tgt = np.asarray(env_i.target).ravel()

        msg_state = {
            "episode_id": 0,
            "t": t_sec,
            "x": float(s[0]),
            "y": float(s[1]),
            "yaw": float(s[2]),
            "v": float(s[3]) if s.size > 3 else 0.0,
            "state": [float(v) for v in s],
        }
        logger.write(f"/agent{i}/state", t_ns, msg_state)

        msg_u = {
            "episode_id": 0,
            "t": t_sec,
            "u_sac": [float(v) for v in np.asarray(a_i).ravel()],
            "delta_fl": float(w_d[0]),
            "delta_fr": float(w_d[1]),
            "delta_rl": float(w_d[2]),
            "delta_rr": float(w_d[3]),
            # wheel speeds if you have them; otherwise omit
            # "v_fl": ..., "v_fr": ..., "v_rl": ..., "v_rr": ...
        }
        logger.write(f"/agent{i}/control", t_ns, msg_u)

        return msg_state, w_d, tgt

    msg1, wd1, tgt1 = log_agent(1, env_agent1, a1, r1, info1)
    msg2, wd2, tgt2 = log_agent(2, env_agent2, a2, r2, info2)
    msg3, wd3, tgt3 = log_agent(3, env_agent3, a3, r3, info3)

    # --------- Write one combined SceneUpdate for all agents + all targets ---------
    entities = []
    entities += mu.make_vehicle_scene_update(
        x=msg1["x"], y=msg1["y"], yaw=msg1["yaw"],
        wheel_delta=wd1, target=tgt1, L=float(config.L), W=float(config.W),
        prefix="agent1_"
    )["entities"]
    entities += mu.make_vehicle_scene_update(
        x=msg2["x"], y=msg2["y"], yaw=msg2["yaw"],
        wheel_delta=wd2, target=tgt2, L=float(config.L), W=float(config.W),
        prefix="agent2_"
    )["entities"]
    entities += mu.make_vehicle_scene_update(
        x=msg3["x"], y=msg3["y"], yaw=msg3["yaw"],
        wheel_delta=wd3, target=tgt3, L=float(config.L), W=float(config.W),
        prefix="agent3_"
    )["entities"]

    logger.write("/scene", t_ns, {"entities": entities})


    if (term1 or trunc1) and (term2 or trunc2) and (term3 or trunc3):
        break

logger.close()
