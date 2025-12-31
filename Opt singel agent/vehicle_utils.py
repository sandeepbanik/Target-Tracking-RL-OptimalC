import time
import logging
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import pdb
import math
import config
import copy


# -----------------------------
# Helpers: trig + ackermann math
# -----------------------------
def cot(x):
    return 1.0 / np.tan(x)

def arccot(x):
    # principal value in (-pi/2, pi/2)
    return np.arctan(1.0 / x)


class Vehicle():
    def __init__(self):
        """
        Initialize vehicle parameters from config.py
        """
        self.L = config.L   # Wheelbase length in meters
        self.W = config.W   # Track width in meters
        self.wheel_W = config.wheel_W # Wheel width
        self.wheel_H = config.wheel_H # Wheel height

        self.n = config.n  # Number of state variables (x, y, theta)
        self.max_steering_angle = config.max_steering_angle  # Maximum steering angle in radians
        self.max_velocity = config.max_velocity  # Maximum velocity in m/s  
        self.max_acceleration = config.max_acceleration  # Maximum acceleration in m/s^2
        self.delta_t = config.delta_t  # Time step for simulation in seconds
        self.Tf = config.Tf  # Total simulation time.

        self.reset()

    def reset(self, init_state = np.array([1., 1., 0., 0.])) -> None:
        """
        Reset the vehicle to the initial state.
        Docstring for reset
        
        :param self:
        :param init_state: initial state of the vehicle [x, y, theta, v]
        """
        # Reset the state
        self.state = init_state

        # To update the plot.


    def f_dyn(
            self,
            u: np.ndarray,
            ) -> None:
        
        """
        Define the dynamics of the vehicle.
        :param self:
        :param u: control inputs [v, delta]
        """ 
        dx = math.cos(self.state[2]) * u[0]            # xdot = cos(theta) v
        dy = math.sin(self.state[2]) * u[0]            # ydot = sin(theta) v
        dtheta = (2*u[0] / self.L) * math.tan(u[1])        # thdot = v/L tan(phi)
        dvel = u[0]  # velocity derivative

        self.state = np.array([dx, dy, dtheta, dvel])

    def f_dyn_opc(
            self,
            t,
            x,
            u,
            params=None
            ):
        
        """
        Define the dynamics of the vehicle for optimal control.
        :param self:
        :param t: time
        :param x: current state of the vehicle [x, y, theta, v]
        :param u: control inputs [v, delta]
        """ 

        # Saturate the steering input
        phi = np.clip(u[1], -self.max_steering_angle, self.max_steering_angle)
        vel_c = np.clip(u[0], -self.max_velocity, self.max_velocity)
        dx = math.cos(x[2]) * vel_c            # xdot = cos(theta) v
        dy = math.sin(x[2]) * vel_c            # ydot = sin(theta) v
        dtheta = (2*vel_c / self.L) * math.tan(phi)        # thdot = v/L tan(phi)
        dvel = vel_c  # velocity derivative

        self.state = np.array([dx, dy, dtheta, dvel])

        return self.state
    
    def vehicle_output_opc(t, x):
        """
        Function to get the output of the vehicle.
        """
        return x
        

    def get_state(self,t,x) -> np.ndarray:
        """
        Function to get the current state of the vehicle.
        """
        return self.state.copy()
    
    
    def SNS(self, state,u):
        """
        Symmetric negative steering configuration.
        state - [x,y,theta]
        """
        # Evaluate the front and read wheels angles.
        delta_fr = arccot(cot(u[1]) + self.W/self.L)
        delta_fl = arccot(cot(u[1]) - self.W/self.L)
        v_fr = u[0] * np.tan(u[1])/np.sin(delta_fr)
        v_fl = u[0] * np.tan(u[1])/np.sin(delta_fl)

        delta_rl = -delta_fl
        delta_rr = -delta_fr
        v_rl = v_fl
        v_rr = v_fr

        return delta_fr, delta_fl, delta_rr, delta_rl, v_fr, v_fl, v_rr, v_rl
    