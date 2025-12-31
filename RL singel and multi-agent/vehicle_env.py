# -*- coding: utf-8 -*-
"""
(in gym format) robot environment for training policy
state : x,y,theta,v
action: velocity, steering angle
observation space: relative_distance_to_target, relative_angle_to_target, current_velocity, current_steering_angle
"""
#%%
import numpy as np
import config as config
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config
import helper as hl
import math
import scipy.integrate as spi
import pdb

#%%
class VehicleEnv(gym.Env):
    def __init__(self):
        super(VehicleEnv, self).__init__()
        
        # Vehicle Parameters
        self.L = config.L  # Wheelbase
        self.W = config.W # Track width
        self.dt = config.delta_t # Time step
        
        # Limits
        self.n = config.n  # Number of state variables (x, y, theta)
        self.max_steering_angle = config.max_steering_angle  # Maximum steering angle in radians
        self.max_velocity = config.max_velocity  # Maximum velocity in m/s  
        self.max_acceleration = config.max_acceleration  # Maximum acceleration in m/s^2
        self.delta_t = config.delta_t  # Time step for simulation in seconds
        self.Tf = config.Tf  # Total simulation time.
        self.current_time = 0.0 # Current time.

        # Action Space: [velocity, curvature] normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Observation Space: [rel_x, rel_y, sin(theta), cos(theta), current_v, current_steer]
        # Relative to target
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Internal State
        self.state = np.zeros(self.n)      # [x, y, theta, v]
        self.target = np.zeros(config.n)     # [x, y]
        self.wheel_v = np.zeros(4)    # [FR, FL, RR, RL]
        self.wheel_delta = np.zeros(4) # [FR, FL, RR, RL]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomize a target location.
        self.target = np.array([np.random.uniform(-10.0, 10.0), np.random.uniform(-10.0, 10.0)])
        # Start from any pose within 10m of target (0,0)
        angle = self.np_random.uniform(0, np.pi)
        dist = self.np_random.uniform(15.0, 18.0)
        self.state = np.array([self.target[0] + dist * np.cos(angle) + np.random.rand(), 
                               self.target[1] + dist * np.sin(angle) + np.random.rand(), 
                               self.np_random.uniform(-np.pi, np.pi), 0.0])
        self.current_time = 0.0     # Reset time.
        # Reintialize wheel states.
        self.wheel_v = np.zeros(4)
        self.wheel_delta = np.zeros(4)

        # Reset step count.
        self.current_step = 0
        self.max_steps = config.max_steps
        
        return self._get_obs(),{}

    def _get_obs(self):
        """
        Function to get observation vector.
        :param self: Description
        """ 
        dx = self.target[0] - self.state[0] # X difference to target
        dy = self.target[1] - self.state[1] # Y difference to target
        theta = self.state[2] # Heading angle
        
        # Rotation matrix into robot frame
        rel_x = dx * np.cos(theta) + dy * np.sin(theta)
        rel_y = -dx * np.sin(theta) + dy * np.cos(theta)
        
        return np.array([rel_x, rel_y, np.arctan2(np.sin(theta), np.cos(theta))], dtype=np.float32)
    
    def get_state(self):
        return np.array(self.state, dtype=float).copy()


    def SNS(self, state,u):
        """
        Symmetric negative steering configuration.
        state - [x,y,theta]
        """
        # Evaluate the front wheel angles and velocity.
        delta_fr = hl.arccot(hl.cot(u[1]) + self.W/self.L)
        delta_fl = hl.arccot(hl.cot(u[1]) - self.W/self.L)
        v_fr = u[0] * np.tan(u[1])/np.sin(delta_fr)
        v_fl = u[0] * np.tan(u[1])/np.sin(delta_fl)

        # Evaluate the rear wheel angles and velocity.
        delta_rl = -delta_fl
        delta_rr = -delta_fr
        v_rl = v_fl
        v_rr = v_fr

        return delta_fr, delta_fl, delta_rr, delta_rl, v_fr, v_fl, v_rr, v_rl
    
    def f_dyn_opc(
            self,
            t,
            v,
            sa
            ):
        
        """
        Define the dynamics of the vehicle for optimal control.
        :param self:
        :param t: time
        :param x: current state of the vehicle [x, y, theta, v]
        :param u: control inputs [v, delta]
        """ 
        # Saturate the steering input and velocity
        phi = np.clip(sa, -self.max_steering_angle, self.max_steering_angle)
        vel_c = np.clip(v, -self.max_velocity, self.max_velocity)

        self.state[0] += vel_c * np.cos(self.state[2]) * self.dt
        self.state[1] += vel_c * np.sin(self.state[2]) * self.dt
        self.state[2] += (2*vel_c / self.L) * np.tan(phi) * self.dt
        self.state[3] += np.clip(vel_c*self.dt, 0, self.max_acceleration) # Saturate acceleration
       
        # Update wheel data.
        delta_fr, delta_fl, delta_rr, delta_rl, v_fr, v_fl, v_rr, v_rl = self.SNS(self.state, [v,sa])
        self.wheel_delta[0] = delta_fr
        self.wheel_delta[1] = delta_fl
        self.wheel_delta[2] = delta_rr
        self.wheel_delta[3] = delta_rl

    def step(self, action):
        
        """
        Step the environment by one time step.
        action: [normalized_velocity, normalized_curvature]
        """
        # Discrete time step
        self.current_step += 1


        # actions mapped to  [-1, 1] to physical limits
        target_v = action[0] * self.max_velocity
        target_sa = action[1] * self.max_steering_angle
        
        # Integrate the vehicle dynamics over delta_t (Euler integration).
        self.f_dyn_opc(self.current_time, target_v, target_sa)
        self.current_time += self.dt # Update current time.
        
        # Reward and Termination
        dist = np.linalg.norm(self.state[0:2] - self.target)
        reward = -dist - np.linalg.norm(action)  # Distance + control penalty

        # Termination conditions
        terminated = bool(dist < 0.3)
        truncated = bool(self.current_step >= self.max_steps) 
        
        info = {"distance": dist}
        
            
        return self._get_obs(), reward, terminated, truncated, info

