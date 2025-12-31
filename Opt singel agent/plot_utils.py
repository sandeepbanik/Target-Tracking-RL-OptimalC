import matplotlib.pyplot as plt
import config
from matplotlib.patches import Rectangle
from matplotlib import animation
import numpy as np
import vehicle_utils as vu
import pdb


# -----------------------------
# Helpers: trig + ackermann math
# -----------------------------
def cot(x):
    return 1.0 / np.tan(x)

def arccot(x):
    # principal value in (-pi/2, pi/2)
    return np.arctan(1.0 / x)


def plot_states(t, y, u, yf=None, figure=None):
    plt.figure(figure)
 
    # Plot the xy trajectory
    plt.subplot(3, 1, 1)
    plt.plot(y[0], y[1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    if yf is not None:
        plt.plot(yf[0], yf[1], 'ro')
 
    # Plot the inputs as a function of time
    plt.subplot(3, 1, 2)
    plt.plot(t, u[0])
    plt.xlabel("t [sec]")
    plt.ylabel("velocity [m/s]")
    plt.subplot(3, 1, 3)
    plt.plot(t, u[1])
    plt.xlabel("t [sec]")
    plt.ylabel("steering [rad/s]")

    plt.tight_layout()
    plt.show(block=False)

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s,  c]])

def set_rect_center_angle(rect: Rectangle, center_xy, angle_rad, length, width):
    # Rectangle expects lower-left + angle in degrees about that corner.
    # We set lower-left so that the rectangle is centered at center_xy.
    cx, cy = center_xy
    ll = np.array([-length/2, -width/2])  # local lower-left w.r.t. center
    R = rot2(angle_rad)
    ll_world = np.array([cx, cy]) + R @ ll

    rect.set_xy(ll_world)
    rect.angle = np.degrees(angle_rad)
    rect.set_width(length)
    rect.set_height(width)





def plot_vehicle(vehicle_model, states, controls, opc_time):
    
    """
    Plot function for the vehicle trajectory and animation.
    :param states: [x, y, theta, v] states of the vehicle over time
    """
    # pdb.set_trace()
    x,y,theta,v = states
    v,delta = controls
    tspan = opc_time

    body_L = config.L 
    body_W = config.W

    wheel_W = body_W/4
    wheel_H = body_L/4


    # -----------------------------
    # Plot + animation
    # -----------------------------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.axis('equal')
    ax.grid(True)

    # Trajectory background
    ax.plot(x, y, lw=1.5)
    traj_line, = ax.plot([], [], lw=2)  # animated partial trajectory

    # Body rectangle (we will place it so rear axle is at ~25% of body length from the back)
    body = Rectangle((x[0]-body_L/2, y[0]-body_W/2), body_L, body_W, angle = np.degrees(theta[0]), fill=False, linewidth=2)
    # body = Rectangle((x[100]-body_L/2, y[100]-body_W/2), body_L, body_W, angle = np.degrees(theta[100]), fill=False, linewidth=2)
    ax.add_patch(body)

    # Determine the initial wheel positions.
    d_fr, d_fl, d_rr, d_rl, v_fr, v_fl, v_rr, v_rl = vehicle_model.SNS([x[0], y[0], theta[0]], [0, 0])
    # Wheel rectangles
    wFR = Rectangle((x[0]+body_L/2-wheel_H/2, y[0]-body_W/2-wheel_W/2), wheel_H, wheel_W, angle = np.degrees(d_fr), fill=True, alpha=0.9)
    wFL = Rectangle((x[0]+body_L/2-wheel_H/2, y[0]+body_W/2-wheel_W/2), wheel_H, wheel_W, angle = np.degrees(d_fl), fill=True, alpha=0.9)
    wRR = Rectangle((x[0]-body_L/2-wheel_H/2, y[0]-body_W/2-wheel_W/2), wheel_H, wheel_W, angle = np.degrees(d_rr), fill=True, alpha=0.9)
    wRL = Rectangle((x[0]-body_L/2-wheel_H/2, y[0]+body_W/2-wheel_W/2), wheel_H, wheel_W, angle = np.degrees(d_rl), fill=True, alpha=0.9)
    for w in (wFR, wFL, wRR, wRL):
        ax.add_patch(w)


    def update_v_body(k):
        # Extract vehicle pose
        xr, yr, th = x[k], y[k], theta[k]
        vr, deltar = v[k], delta[k]

        # Update the rectangles patches
        d_fr, d_fl, d_rr, d_rl, v_fr, v_fl, v_rr, v_rl = vehicle_model.SNS([xr, yr, th], [vr, deltar])
        # Car body update.
        set_rect_center_angle(body, (xr, yr), th, body_L, body_W)

        R = rot2(th)
        offsets = {
        'FR': np.array([ body_L/2, -body_W/2]),
        'FL': np.array([ body_L/2,  body_W/2]),
        'RR': np.array([-body_L/2, -body_W/2]),
        'RL': np.array([-body_L/2,  body_W/2])
        }
        
        pos_fr = np.array([xr, yr]) + R @ offsets['FR']
        set_rect_center_angle(wFR, pos_fr, th + d_fr, wheel_H, wheel_W)

        pos_fl = np.array([xr, yr]) + R @ offsets['FL']
        set_rect_center_angle(wFL, pos_fl, th + d_fl, wheel_H, wheel_W)

        pos_rr = np.array([xr, yr]) + R @ offsets['RR']
        set_rect_center_angle(wRR, pos_rr, th + d_rr, wheel_H, wheel_W)

        pos_rl = np.array([xr, yr]) + R @ offsets['RL']
        set_rect_center_angle(wRL, pos_rl, th + d_rl, wheel_H, wheel_W)



        # Partial trajectory
        traj_line.set_data(x[:k+1], y[:k+1])

        return traj_line, body, wFR, wFL, wRR, wRL
    
    ani = animation.FuncAnimation(fig, 
                                  update_v_body, frames=len(tspan), interval=20, blit=True
        )

    plt.show()
    return ani


def plot_vehicle_to_target(vehicle_model, states, controls, target, time_span):
    
    """
    Plot function for the vehicle trajectory to target and animation.
    :param states: [x, y, theta, v] states of the vehicle over time
    """
    x,y,theta,v = states
    v,delta = controls
    tspan = time_span
    target_x, target_y = target

    body_L = config.L 
    body_W = config.W

    wheel_W = body_W/4
    wheel_H = body_L/4


    # -----------------------------
    # Plot + animation
    # -----------------------------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.axis('equal')
    ax.grid(True)

    # Trajectory background
    ax.plot(x, y, lw=1.5)
    traj_line, = ax.plot([], [], lw=2)  # animated partial trajectory

    # Draw the target point
    ax.plot(target_x, target_y, 'rx', markersize=10, label='Target')


    # Body rectangle (we will place it so rear axle is at ~25% of body length from the back)
    body = Rectangle((x[0]-body_L/2, y[0]-body_W/2), body_L, body_W, angle = np.degrees(theta[0]), fill=False, linewidth=2)
    # body = Rectangle((x[100]-body_L/2, y[100]-body_W/2), body_L, body_W, angle = np.degrees(theta[100]), fill=False, linewidth=2)
    ax.add_patch(body)

    # Determine the initial wheel positions.
    d_fr, d_fl, d_rr, d_rl, v_fr, v_fl, v_rr, v_rl = vehicle_model.SNS([x[0], y[0], theta[0]], [0, 0])
    # Wheel rectangles
    wFR = Rectangle((x[0]+body_L/2-wheel_H/2, y[0]-body_W/2-wheel_W/2), wheel_H, wheel_W, angle = np.degrees(d_fr), fill=True, alpha=0.9)
    wFL = Rectangle((x[0]+body_L/2-wheel_H/2, y[0]+body_W/2-wheel_W/2), wheel_H, wheel_W, angle = np.degrees(d_fl), fill=True, alpha=0.9)
    wRR = Rectangle((x[0]-body_L/2-wheel_H/2, y[0]-body_W/2-wheel_W/2), wheel_H, wheel_W, angle = np.degrees(d_rr), fill=True, alpha=0.9)
    wRL = Rectangle((x[0]-body_L/2-wheel_H/2, y[0]+body_W/2-wheel_W/2), wheel_H, wheel_W, angle = np.degrees(d_rl), fill=True, alpha=0.9)
    for w in (wFR, wFL, wRR, wRL):
        ax.add_patch(w)


    def update_v_body(k):
        # Extract vehicle pose
        xr, yr, th = x[k], y[k], theta[k]
        vr, deltar = v[k], delta[k]

        # Update the rectangles patches
        d_fr, d_fl, d_rr, d_rl, v_fr, v_fl, v_rr, v_rl = vehicle_model.SNS([xr, yr, th], [vr, deltar])
        # Car body update.
        set_rect_center_angle(body, (xr, yr), th, body_L, body_W)

        R = rot2(th)
        offsets = {
        'FR': np.array([ body_L/2, -body_W/2]),
        'FL': np.array([ body_L/2,  body_W/2]),
        'RR': np.array([-body_L/2, -body_W/2]),
        'RL': np.array([-body_L/2,  body_W/2])
        }
        
        pos_fr = np.array([xr, yr]) + R @ offsets['FR']
        set_rect_center_angle(wFR, pos_fr, th + d_fr, wheel_H, wheel_W)

        pos_fl = np.array([xr, yr]) + R @ offsets['FL']
        set_rect_center_angle(wFL, pos_fl, th + d_fl, wheel_H, wheel_W)

        pos_rr = np.array([xr, yr]) + R @ offsets['RR']
        set_rect_center_angle(wRR, pos_rr, th + d_rr, wheel_H, wheel_W)

        pos_rl = np.array([xr, yr]) + R @ offsets['RL']
        set_rect_center_angle(wRL, pos_rl, th + d_rl, wheel_H, wheel_W)



        # Partial trajectory
        traj_line.set_data(x[:k+1], y[:k+1])

        return traj_line, body, wFR, wFL, wRR, wRL
    
    ani = animation.FuncAnimation(fig, 
                                  update_v_body, frames=len(tspan), interval=20, blit=True
        )

    plt.show()
    return ani