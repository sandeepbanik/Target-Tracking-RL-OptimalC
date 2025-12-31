import vehicle_utils as vu
import numpy as np
import math
import config
import control as ct
import control.optimal as obc
import time
import copy
import pdb

class optimal_control:
    def __init__(self, vehicle_model: vu.Vehicle):
        
        """
        Intialize the Optimal Control for the vehicle
        """
        self.vehicle_model = vehicle_model
        self.n_dim = config.n  # Number of state variables
        self.m_dim = config.m  # Number of control inputs

        self.Tf = config.Tf  # Total simulation time.
        self.delta_t = config.delta_t  # Time step for simulation in seconds

        self.Q = np.diag([config.Q_x, config.Q_y, config.Q_theta, config.Q_v])  # State cost weights
        self.R = np.diag([config.R_velocity, config.R_steering])  # Input cost weights

        self.x_f = config.x_f  # Target x position in meters
        self.y_f = config.y_f   # Target y position in meters
        self.theta_f = config.theta_f  # Target orientation in radians
        self.v_f = config.v_f   # Target velocity in m/s

        self.Tf = config.Tf  # Total simulation time.
        self.N_sim_opc = config.N_sim_opc  # Number of simulation steps for optimal control

        self.tspan = np.linspace(0, self.Tf, self.N_sim_opc, endpoint=True)  # Time span for simulation

        self.x0 = vehicle_model.state  # Initial state
        self.xf = np.array([self.x_f, self.y_f, self.theta_f, self.v_f])  # Target state
        self.u0 = np.array([0.0, 0.0])  # Initial control inputs
        self.init_guess = (np.array([self.x0 + (self.xf - self.x0) * time/self.Tf for time in self.tspan]).transpose(), np.outer(self.u0, np.ones_like(self.tspan)))

        self.opt_problem_setup()

    def opt_problem_setup(self):
        """
        Setup the optimal control problem.
        """
        # pdb.set_trace()
        # Define the vehicle steering dynamics as an input/output system
        self.vehicle = ct.NonlinearIOSystem(updfcn=self.vehicle_model.f_dyn_opc,
                                        outfcn=self.vehicle_model.vehicle_output_opc,
                                        states=self.n_dim, name='vehicle',
                                        inputs=('v', 'phi'),
                                        outputs=('x', 'y', 'theta','v'))

        self.quad_cost = obc.quadratic_cost(self.vehicle, self.Q, self.R, x0=self.xf, u0=self.u0)
        
        self.constraints = [obc.input_range_constraint(self.vehicle, [-self.vehicle_model.max_velocity, -self.vehicle_model.max_steering_angle],
                                                        [self.vehicle_model.max_velocity, self.vehicle_model.max_steering_angle]) ]


    def compute_opc(self):
        """
        Compute the optimal control inputs using Optimal Control library.
        """

        start_time = time.process_time() # Start time measurement

        self.opt_u = obc.solve_ocp(self.vehicle,
                              self.tspan,
                              self.x0, 
                              self.quad_cost,
                              constraints=self.constraints,
                              initial_guess=self.init_guess,    
                              log=True,
            # minimize_method='trust-constr',
            # minimize_options={'finite_diff_rel_step': 0.01},
        )
        end_time = time.process_time()
        print("* Total time = %5g seconds\n" % (end_time - start_time))
        
        return self.opt_u

class DWAController_CollisionAvoidance:
    def __init__(self):
        """
        Intialize the DWA Controller for Collision Avoidance
        """
        self.m_dim = config.m  # Number of control inputs
        self.n_dim = config.n  # Number of state variables

        self.T = config.T  # Time horizon for DWA in seconds
        self.delta_t = config.delta_t  # Time step for simulation in seconds
        self.N_sim = config.N_sim  # Number of simulation steps

        # Intialize control inputs
        self.u_prev = np.zeros((self.N_sim, self.m_dim))

    def compute_control(self,
                        observed_state: np.ndarray) -> np.ndarray:
        """
        Compute the control inputs using DWA with collision avoidance.

        :param self:
        :param observed_state: observed state of the vehicle [x, y, theta, v]
        """

        # Previous control inputs
        u_prev = self.u_prev

        # Current state
        x = observed_state


