# -*- coding: utf-8 -*-
"""
Contains helper functions to code bicycle model
"""
#%%
import numpy as np
from matplotlib.patches import Rectangle
import config
#%%
def cot(x):
    return 1.0 / np.tan(x)

def arccot(x):
    # principal value in (-pi/2, pi/2)
    return np.arctan(1.0 / x)

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s,  c]])

# Define the bicycle model dynamics
def bicycle_model(t, state, v, delta):
    x, y, theta = state
    dxdt = v * np.cos(theta)
    dydt = v * np.sin(theta)
    dthetadt = 2*v / config.L * np.tan(delta)
    return [dxdt, dydt, dthetadt]


def SNS(state,v,delta):
    """
    Symmetric negative steering configuration.
    state - [x,y,theta]
    """
    # Evaluate the front and read wheels angles.
    delta_fr = arccot(cot(delta) + config.W/config.L)
    delta_fl = arccot(cot(delta) - config.W/config.L)
    v_fr = v * np.tan(delta)/np.sin(delta_fr)
    v_fl = v * np.tan(delta)/np.sin(delta_fl)

    delta_rl = -delta_fl
    delta_rr = -delta_fr
    v_rl = v_fl
    v_rr = v_fr

    return delta_fr, delta_fl, delta_rr, delta_rl, v_fr, v_fl, v_rr, v_rl


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
    
def set_rect_center_angle_v2(rect: Rectangle, center_xy, angle_rad):
    # Rectangle expects lower-left + angle in degrees about that corner.
    
    rect.set_xy(center_xy)
    rect.angle = np.degrees(angle_rad)
    


