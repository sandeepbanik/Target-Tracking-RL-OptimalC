import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import plot_utils as p_utl

import vehicle_utils as vu
import control_utils as cu

import config

# Define the vehicle model for the simulation.
SNS_vehicle = vu.Vehicle()

# Define the optimal control for the vehicle.
SNS_opt_control = cu.optimal_control(SNS_vehicle)

# Solve the control.
opt_u = SNS_opt_control.compute_opc()

# Plot the vehicle
p_utl.plot_vehicle(SNS_vehicle, opt_u.states, opt_u.inputs, opt_u.time)

# Plot the state.
p_utl.plot_states(opt_u.time, opt_u.states, opt_u.inputs)