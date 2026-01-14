import numpy as np
from scipy.integrate import solve_ivp
import control as ct

from config_8thruster import *
from state_space import *
from lqr import *

"""
state = [
    x position (m)
    y position (m)
    z position [depth] (m)
    roll (rad)
    pitch (rad)
    yaw (rad)
    x velocity (m/s)
    y velocity (m/s)
    z velocity (m/s)
    roll velocity (rad/s)
    pitch velocity (rad/s)
    yaw velocity (rad/s)
]
"""

def main():
    state_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    target_state = np.array([1, 2, 3, 0, 0, 1.57, 0, 0, 0, 0, 0, 0])

    Q = np.array([1, 1, 4, 4, 1, 1, 3, 3, 3, 4, 4, 4])
    R = 0.001

    # simulation parameter
    t_lower = 0     # lower bound (s)
    t_upper = 60*2   # upper bound (s)
    h = 2           # time step (s)
    t_span = np.arange(t_lower, t_upper + h, h)

    # state space
    ss = StateSpace()

    # lqr
    lqr = LQR(state_0, target_state, ss.df_dstate_funct, ss.df_dcontrol, ss.G, ss.thrust_allocation, Q, R)

    print("Wait until completed...")
    sol = solve_ivp(lqr.simulated_step, [t_lower, t_upper], state_0, t_eval=t_span)


    import matplotlib.pyplot as plt
    state = sol.y.T

    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(t_span, state[:, :6])
    plt.title('Robot Pose')
    plt.xlabel('time (s)')
    plt.ylabel('(m) or (rad)')
    plt.legend(['x','y','z','roll','pitch','yaw'])
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(t_span, state[:, 6:12])
    plt.title('Robot Velocity')
    plt.xlabel('time (s)')
    plt.ylabel('(m/s) or (rad/s)')
    plt.legend(['u','v','w','p','q','r'])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
