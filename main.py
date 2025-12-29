import numpy as np
from scipy.integrate import solve_ivp
import control as ct

from config import *
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

def loop(t, state, lqr):
    A, B, du = lqr.run(state)
    return (A @ state + B @ du).flatten()

def main():
    state_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    target_state = np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Q = np.array([1, 1, 4, 4, 1, 4, 3, 3, 3, 4, 4, 4])
    R = 0.001

    # simulation parameter
    t_lower = 0     # lower bound (s)
    t_upper = 60*3   # upper bound (s)
    h = 2           # time step (s)
    t_span = np.arange(t_lower, t_upper + h, h)

    # state space
    ss = StateSpace()

    # lqr
    lqr = LQR(state_0, target_state, ss.df_dstate_funct, ss.df_dcontrol, ss.G, ss.thrust_allocation, Q, R)

    sol = solve_ivp(loop, [t_lower, t_upper], state_0, t_eval=t_span, args=(lqr,), vectorized=False)


    import matplotlib.pyplot as plt
    state = sol.y.T  # shape: (time_steps, 12)

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
