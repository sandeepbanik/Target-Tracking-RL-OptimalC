import numpy as np


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
