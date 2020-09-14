import numpy as np

from model import Pendulum
from dlqr import DLQR
from simulation import EmbeddedSimEnvironment

# Create pendulum and controller objects
PART1_DENUG = False
PART2_DEBUG = False
PART3_DEBUG = True
pendulum = Pendulum()
rho = 1
# Get the system discrete-time dynamics
A, B, Bw, C = pendulum.get_discrete_system_matrices_at_eq()
# print(A.shape)
ctl = DLQR(A, B, C)
# Get control gains
K, P = ctl.get_lqr_gain(Q = np.diag([1.0 / (8.75 ** 2), 1.0 / (2.67 ** 2),\
                                     1.0 / (0.189 ** 2), 1.0 / (0.32 ** 2)]), R = 1.0 / (1.57 ** 2)) # From Assignment1, the max u is 1.57

# Get feedforward gain
lr = ctl.get_feedforward_gain(K)
if (PART1_DENUG == True):
    # Part I - no disturbance
    sim_env = EmbeddedSimEnvironment(model=pendulum, 
                                    dynamics=pendulum.discrete_time_dynamics,
                                    controller=ctl.feedfwd_feedback,
                                    time = 20)
    sim_env.set_window(20)
    t, y, u = sim_env.run([0,0,0,0])

    # Part I - with disturbance
    pendulum.enable_disturbance(w=0.01)  
    sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
                                    dynamics=pendulum.discrete_time_dynamics,
                                    controller=ctl.feedfwd_feedback,
                                    time = 20)
    sim_env_with_disturbance.set_window(20)
    t, y, u = sim_env_with_disturbance.run([0,0,0,0])

if (PART2_DEBUG == True):
    ### Part II
    Ai, Bi, Bwi, Ci = pendulum.get_augmented_discrete_system()
    # print(Ai.shape)
    ctl.set_system(Ai, Bi, Ci)
    K, P = ctl.get_lqr_gain(Q = np.diag([1, 50, 1, 1000, 1]), 
                            R = 1.0/4) 
    # lr = ctl.get_feedforward_gain(K)
    # Get feed-forward gain              
    ctl.set_lr(lr)     

    pendulum.enable_disturbance(w=0.01)  
    sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
                                    dynamics=pendulum.pendulum_augmented_dynamics,
                                    controller=ctl.lqr_ff_fb_integrator,
                                    time = 20)
    sim_env_with_disturbance.set_window(20)
    t, y, u = sim_env_with_disturbance.run([0,0,0,0,0])

if (PART3_DEBUG == True):
    ### Part III
    # Output feedback
    Ai, Bi, Bwi, Ci = pendulum.get_augmented_discrete_system()
    ctl.set_system(Ai, Bi, Ci)
    K, P = ctl.get_lqr_gain(Q = np.diag([1.0 / (8.75 ** 2), 1.0 / (2.67 ** 2),\
                                     1.0 / (0.189 ** 2), 1.0 / (0.32 ** 2), 1]), R = 1.0 / 4)
    # C = np.array([[1,0,0,0]])
    C = np.array([[1,0,0,0],
                [0,0,1,0]])

    Qp = np.eye(4)
    # Qp = 50 * np.eye(4)

    Rn = np.eye(np.size(C,0))
    pendulum.set_kf_params(C,Qp,Rn)
    pendulum.init_kf()
    pendulum.enable_disturbance(w=0.01)  

    sim_env_with_disturbance_estimated = EmbeddedSimEnvironment( model=pendulum, 
                                                                 dynamics=pendulum.pendulum_augmented_dynamics,
                                                                 controller=ctl.lqr_ff_fb_integrator,
                                                                 time = 20)
    sim_env_with_disturbance_estimated.set_estimator(True)
    sim_env_with_disturbance_estimated.set_window(20)
    t, y, u = sim_env_with_disturbance_estimated.run([0,0,0,0,0])