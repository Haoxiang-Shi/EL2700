import numpy as np

from model import Pendulum
from dlqr import DLQR
from simulation import EmbeddedSimEnvironment

ENABLE_AUGMENTED = True # False --> Part 1; True --> Part 2 & 3
ENABLE_DSTR = True # True --> Enable distrubance

PART2_DEBUG = False
PART3_DEBUG = True

# Create pendulum and controller objects
pendulum = Pendulum()
# Get the system discrete-time dynamics
A, B, Bw, C = pendulum.get_discrete_system_matrices_at_eq()
ctl = DLQR(A, B, C)
# Set parameter rho
rho = 1
# Get control gains
K, P = ctl.get_lqr_gain(Q= np.diag([1.0/100, 1.0/25, 1.0/0.03, 1.0/0.5]), 
                    R=rho*1.0/4)

# Get feeforward gain
lr = ctl.get_feedforward_gain(K)

if ENABLE_AUGMENTED is False:
    
# Part I - no disturbance
    if ENABLE_DSTR is False:
        sim_env = EmbeddedSimEnvironment(model=pendulum, 
                                    dynamics=pendulum.discrete_time_dynamics,
                                    controller=ctl.feedfwd_feedback,
                                    time = 10)
        sim_env.set_window(10)
        t, y, u = sim_env.run([0,0,0,0])

    # Part I - with disturbance
    if ENABLE_DSTR is True:
        pendulum.enable_disturbance(w=0.01)  
        sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
                                    dynamics=pendulum.discrete_time_dynamics,
                                    controller=ctl.feedfwd_feedback,
                                    time = 20)
        sim_env_with_disturbance.set_window(20)
        t, y, u = sim_env_with_disturbance.run([0,0,0,0])

if ENABLE_AUGMENTED is True:
    ### Part II
    if PART2_DEBUG is True:
        Ai, Bi, Bwi, Ci = pendulum.get_augmented_discrete_system()
        ctl.set_system(Ai, Bi, Ci)
        K, P = ctl.get_lqr_gain(Q= np.diag([1.0/100, 1.0/25, 20*1.0/0.03, 1.0/0.5, 1]), 
                            R= rho*1/4)

        # Get feeforward gain              
        ctl.set_lr(lr)     

        pendulum.enable_disturbance(w=0.01)  
        sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
                                    dynamics=pendulum.pendulum_augmented_dynamics,
                                    controller=ctl.lqr_ff_fb_integrator,
                                    time = 10)
        sim_env_with_disturbance.set_window(10)
        t, y, u = sim_env_with_disturbance.run([0,0,0,0,0])

    ### Part III
    if PART3_DEBUG is True:
    # Output feedback
        Ai, Bi, Bwi, Ci = pendulum.get_augmented_discrete_system()
        ctl.set_system(Ai, Bi, Ci)
        K, P = ctl.get_lqr_gain(Q= np.diag([1.0/100, 1.0/25, 20*1.0/0.03, 1.0/0.5, 1]), 
                            R= rho*1/4)

        # Get feeforward gain              
        ctl.set_lr(lr)     

        pendulum.enable_disturbance(w=0.01)

        C = np.array([[1,0,0,0]])
        C = np.array([[1,0,0,0],
                [0,0,1,0]])

        Qp = np.eye(4)
        Rn = np.eye(np.size(C,0))
        pendulum.set_kf_params(C,Qp,Rn)
        pendulum.init_kf()

        sim_env_with_disturbance_estimated = EmbeddedSimEnvironment(model=pendulum, 
                                    dynamics=pendulum.pendulum_augmented_dynamics,
                                    controller=ctl.lqr_ff_fb_integrator,
                                    time = 10)
        sim_env_with_disturbance_estimated.set_estimator(True)
        sim_env_with_disturbance_estimated.set_window(10)
        t, y, u = sim_env_with_disturbance_estimated.run([0,0,0,0,0])