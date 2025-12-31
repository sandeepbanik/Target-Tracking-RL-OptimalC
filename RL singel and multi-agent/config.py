import math

# Parameters for the bicycle model
L = 2.0  # Wheelbase length in meters
n = 4 # Number of state variables (x, y, theta, v)
m = 2 # Number of control inputs (v, delta)
W = 1.0  # Track width in meters
wheel_W = W/4 # Wheel width
wheel_H = L/4 # Wheel height

# Parameters for the simulation and control
max_steering_angle = math.radians(30)  # Maximum steering angle in radians
max_velocity = 20.0  # Maximum velocity in m/s
max_acceleration = 2.0  # Maximum acceleration in m/s^2
delta_t = 0.05  # Time step for ODE simulation in seconds
N_sim = 10 # Number of simulation steps
T = delta_t*N_sim # Time horizon for DWA in seconds.
N_sim_opc = 50 # Number of simulation steps for optimal control

Tf = 5.0  # Total simulation time.

# Define cost function weights.
Q_x = 1.0  # State cost weights
Q_y = 1.0
Q_theta = 0.1
Q_v = 0.1

R_steering = 1.0  # Input cost weights
R_velocity = 1.0 


safe_r = 0.2 # Safety radius for collision avoidance.

# Target configuration
x_f = 10.0  # Target x position in meters
y_f = 10.5   # Target y position in meters
theta_f = 0.0  # Target orientation in radians
v_f = 0.0   # Target velocity in m/s
n_target = 2 # Target x and y state

## RL hyperparameters
learning_rate = 1e-4
buffer_size = int(1e6)
batch_size = 256
total_timesteps = 200000
max_steps = 1000

n_agent = 3