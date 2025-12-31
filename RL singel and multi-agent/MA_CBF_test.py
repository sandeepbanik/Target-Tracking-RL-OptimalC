# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 11:44:44 2025

@author: sande
"""

import numpy as np
import vehicle_env as vehicle
from stable_baselines3 import SAC

 


def cbf_filter_velocity_only_vehicleenv(env, action, other_positions_xy, d_min=1.5, kappa=2.0, eps=1e-9):
    """
    CBF function acting only on the velocity command of VehicleEnv.
    Assumptions:
    Safety function for any agent j:
        h_j = ||p - p_j||^2 - d_min^2
    CBF condition (continuous-time form):
        hdot_j + kappa * h_j >= 0
        Agent position is assumed to be constant. 

    Bicycle kinematics with only velocity control:
        p_dot = [v cos(theta), v sin(theta)]
    
    =>  hdot_j = 2 (p - p_j)^T p_dot = 2 v (p - p_j)^T [cos(theta), sin(theta)] = a_j * v

    This yields linear constraint in physical v (m/s):
        a_j * v + kappa * h_j >= 0

    Based on the constraints and the sign of a_j, we clamp the velocity.
    """
    a = np.asarray(action, dtype=float).copy()

    # Current state
    x, y, theta = float(env.state[0]), float(env.state[1]), float(env.state[2])
    p = np.array([x, y], dtype=float)
    dir_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    # Map normalized velocity to physical velocity (m/s)
    v_cmd = float(a[0]) * float(env.max_velocity)

    v_safe = v_cmd
    others = np.asarray(other_positions_xy, dtype=float).reshape(-1, 2)
    for p_j in others:
        diff = p - p_j
        h = float(diff @ diff - d_min**2)

        # If already violating, brake (best-effort)
        if h <= 0.0:
            v_safe = min(v_safe, 0.0)
            continue

        coeff = float(2.0 * (diff @ dir_vec))  # multiplies v in hdot

        # Constraint: coeff * v + kappa * h >= 0
        # If coeff < 0, forward motion (v>0) decreases h, so we impose an upper bound on v.
        if coeff < -eps:
            v_upper = (kappa * h) / (-coeff)
            v_safe = min(v_safe, v_upper)

    # Clamp to physical limits then back to normalized [-1,1]
    v_safe = float(np.clip(v_safe, -env.max_velocity, env.max_velocity))
    a[0] = float(np.clip(v_safe / env.max_velocity, -1.0, 1.0))
    # a[1] = float(np.clip(a[1], -1.0, 1.0))
    return a


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

history = []
max_steps = 1000

# CBF parameters
d_min = 2.0
kappa = 1.0

print("Simulating 3-agent episode (shared SAC policy + velocity-only CBF collision avoidance)...")
for t in range(max_steps):
    # Snapshot current positions for symmetric filtering
    p1 = np.array([env_agent1.state[0], env_agent1.state[1]], dtype=float)
    p2 = np.array([env_agent2.state[0], env_agent2.state[1]], dtype=float)
    p3 = np.array([env_agent3.state[0], env_agent3.state[1]], dtype=float)

    # 1) Predict actions from each policy (synchronous)
    a1, _ = model_agent1.predict(obs1, deterministic=True)
    a2, _ = model_agent2.predict(obs2, deterministic=True)
    a3, _ = model_agent3.predict(obs3, deterministic=True)

    # 2) CBF filter (each agent treats the other agents as obstacles)
    a1 = cbf_filter_velocity_only_vehicleenv(env_agent1, a1, other_positions_xy=[p2, p3], d_min=d_min, kappa=kappa)
    a2 = cbf_filter_velocity_only_vehicleenv(env_agent2, a2, other_positions_xy=[p1, p3], d_min=d_min, kappa=kappa)
    a3 = cbf_filter_velocity_only_vehicleenv(env_agent3, a3, other_positions_xy=[p1, p2], d_min=d_min, kappa=kappa)

    # 3) Step envs (actions computed from the same-time snapshot)
    obs1, r1, term1, trunc1, info1 = env_agent1.step(a1)
    obs2, r2, term2, trunc2, info2 = env_agent2.step(a2)
    obs3, r3, term3, trunc3, info3 = env_agent3.step(a3)

    history.append({
        "t": t,
        "agent1": {"state": env_agent1.state.copy(), "target": env_agent1.target.copy(), "action": np.array(a1, float), "reward": float(r1), "dist": float(info1.get("distance", np.nan)), "wheel_delta":env_agent1.wheel_delta.copy()},
        "agent2": {"state": env_agent2.state.copy(), "target": env_agent2.target.copy(), "action": np.array(a2, float), "reward": float(r2), "dist": float(info2.get("distance", np.nan)), "wheel_delta":env_agent1.wheel_delta.copy()},
        "agent3": {"state": env_agent3.state.copy(), "target": env_agent3.target.copy(), "action": np.array(a3, float), "reward": float(r3), "dist": float(info3.get("distance", np.nan)), "wheel_delta":env_agent1.wheel_delta.copy()},
    })

    # Stop if any agent finishes or times out (change if you want "all done")
    if (term1 or trunc1) and (term2 or trunc2) and (term3 or trunc3):
        break

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from matplotlib.animation import PillowWriter


def rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def set_rect(rect, center, angle, length, width):
    R = rot2(angle)
    ll_local = np.array([-length/2, -width/2], dtype=float)
    ll_world = center + R @ ll_local
    rect.set_xy(ll_world)
    rect.angle = np.degrees(angle)

# Pick any env for geometry parameters (assumed identical)
env_ref = env_agent1
L, W = env_ref.L, env_ref.W
wheel_H, wheel_W = L/4, W/4

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-8, 12)
ax.set_ylim(-5, 12)
ax.set_aspect('equal')
ax.grid(True)

# Draw targets (each agent may have its own target; if same, markers overlap)
t1 = history[0]["agent1"]["target"]
t2 = history[0]["agent2"]["target"]
t3 = history[0]["agent3"]["target"]
ax.plot(t1[0], t1[1], 'rx', markersize=10, label='Target 1')
ax.plot(t2[0], t2[1], 'mx', markersize=10, label='Target 2')
ax.plot(t3[0], t3[1], 'yx', markersize=10, label='Target 3')

# --- Create patches per agent ---
def make_agent_patches(color_body):
    body = Rectangle((0, 0), L, W, fill=False, lw=2, color=color_body)
    wFR = Rectangle((0, 0), wheel_H, wheel_W, fill=True, color='black')
    wFL = Rectangle((0, 0), wheel_H, wheel_W, fill=True, color='black')
    wRR = Rectangle((0, 0), wheel_H, wheel_W, fill=True, color='black')
    wRL = Rectangle((0, 0), wheel_H, wheel_W, fill=True, color='black')
    for p in [body, wFR, wFL, wRR, wRL]:
        ax.add_patch(p)
    return body, (wFR, wFL, wRR, wRL)

body1, wheels1 = make_agent_patches('blue')
body2, wheels2 = make_agent_patches('green')
body3, wheels3 = make_agent_patches('orange')

traj1, = ax.plot([], [], 'b--', alpha=0.5, label='Traj 1')
traj2, = ax.plot([], [], 'g--', alpha=0.5, label='Traj 2')
traj3, = ax.plot([], [], 'orange', linestyle='--', alpha=0.5, label='Traj 3')

offsets = [
    np.array([ L/2, -W/2], dtype=float),  # FR
    np.array([ L/2,  W/2], dtype=float),  # FL
    np.array([-L/2, -W/2], dtype=float),  # RR
    np.array([-L/2,  W/2], dtype=float),  # RL
]

def update(k):
    h = history[k]

    def update_one(agent_key, body, wheels, traj):
        st = np.asarray(h[agent_key]["state"], dtype=float)
        xr, yr, th = float(st[0]), float(st[1]), float(st[2])
        wd = np.asarray(h[agent_key].get("wheel_delta"), dtype=float)

        R = rot2(th)

        # body
        set_rect(body, np.array([xr, yr], dtype=float), th, L, W)

        # wheels
        for i in range(4):
            pos = np.array([xr, yr], dtype=float) + R @ offsets[i]
            set_rect(wheels[i], pos, th + float(wd[i]), wheel_H, wheel_W)

        # trajectory (use history up to k)
        xs = [step[agent_key]["state"][0] for step in history[:k+1]]
        ys = [step[agent_key]["state"][1] for step in history[:k+1]]
        traj.set_data(xs, ys)

        return [traj, body, *wheels]

    artists = []
    artists += update_one("agent1", body1, wheels1, traj1)
    artists += update_one("agent2", body2, wheels2, traj2)
    artists += update_one("agent3", body3, wheels3, traj3)
    return artists

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=50, blit=True)
# Save as GIF
gif_path = "three_agent.gif"
writer = PillowWriter(fps=20)   # fps = 1000 / interval ≈ 20
ani.save(gif_path, writer=writer)
plt.legend()
plt.show()
