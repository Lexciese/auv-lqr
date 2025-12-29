import sympy as sym
from sympy import sin, cos, tan, Derivative
import numpy as np

from config import *

# Fully symbolic non-linear state space model.
# The model is later used in to simulate the dynamics of the robot.
class StateSpace:
    def __init__(self):
        # state and control input
        self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.u, self.v, self.w, self.p, self.q, self.r, self.du0, self.du1, self.du2, self.du3, self.du4, self.du5, self.du6, self.du7 = sym.symbols("x y z roll pitch yaw u v w p q r du0 du1 du2 du3 du4 du5 du6 du7")

        # Mrb matrix
        self.mass, self.Ix, self.Iy, self.Iz, self.Ixy, self.Ixz, self.Iyz, self.mzg = sym.symbols("mass Ix Iy Iz Ixy Ixz Iyz mzg")

        # Ma matrix
        self.Xu_dot, self.Yv_dot, self.Zw_dot, self.Kp_dot, self.Mq_dot, self.Nr_dot, self.Xq_dot, self.Yp_dot = sym.symbols("Xu_dot Yv_dot Zw_dot Kp_dot Mq_dot Nr_dot Xq_dot Yp_dot")

        # Damping matrices
        self.Xu, self.Xuu, self.Yv, self.Yvv, self.Zw, self.Zww, self.Kp, self.Kpp, self.Mq, self.Mqq, self.Nr, self.Nrr = sym.symbols("Xu Xuu Yv Yvv Zw Zww Kp Kpp Mq Mqq Nr Nrr")

        # G matrix
        self.x, self.gy, self.gz, self.bx, self.by, self.bz, self.gravity, self.radius, self.water_density = sym.symbols("x gy gz bx by bz gravity radius water_density")

        self.pose = sym.Matrix([[self.x, self.y, self.z, self.roll, self.pitch, self.yaw]])
        self.vel  = sym.Matrix([[self.u, self.v, self.w, self.p, self.q, self.r]])

        self.state = sym.zeros(12, 1)
        self.state[0:6, 0]  = self.pose.T
        self.state[6:12, 0] = self.vel.T

        self.du = sym.Matrix([[self.du0, self.du1, self.du2, self.du3, self.du4, self.du5]])

        self.gravity_center = sym.Matrix([[gx, gy, gz]])
        self.buoyancy_center = sym.Matrix([[bx, by, bz]])

        self.Mrb = sym.Matrix([[self.mass,   0.0,        0.0,        0.0,        self.mzg,   0.0],
                                      [0.0,         self.mass,  0.0,        -self.mzg,  0.0,        0.0],
                                      [0.0,         0.0,        self.mass,  0.0,        0.0,        0.0],
                                      [0.0,         -self.mzg,  0.0,        self.Ix,    self.Ixy,   self.Ixz],
                                      [self.mzg,    0.0,        0.0,        self.Ixy,   self.Iy,    self.Iyz],
                                      [0.0,         0.0,        0.0,        self.Ixz,   self.Iyz,   self.Iz]])

        self.Ma = sym.Matrix([[self.Xu_dot,  0.0,            0.0,            0.0,            self.Xq_dot,    0.0],
                                     [0.0,          self.Yv_dot,    0.0,            self.Yp_dot,    0.0,            0.0],
                                     [0.0,          0.0,            self.Zw_dot,    0.0,            0.0,            0.0],
                                     [0.0,          self.Yp_dot,    0.0,            self.Kp_dot,    0.0,            0.0],
                                     [self.Xq_dot,       0.0,            0.0,            0.0,            self.Mq_dot,    0.0],
                                     [0.0,          0.0,            0.0,            0.0,            0.0,            self.Nr_dot]])

        self.linear_damping = sym.Matrix([[-self.Xu,    0.0,     0.0,    0.0,    0.0,   0.0],
                                                 [0.0,    -self.Yv,     0.0,    0.0,    0.0,   0.0],
                                                 [0.0,    0.0,     -self.Zw,    0.0,    0.0,   0.0],
                                                 [0.0,    0.0,     0.0,    -self.Kp,    0.0,   0.0],
                                                 [0.0,    0.0,     0.0,    0.0,    -self.Mq,   0.0],
                                                 [0.0,    0.0,     0.0,    0.0,    0.0,   -self.Nr]])

        self.quadratic_damping = sym.Matrix([[-self.Xuu,    0.0,     0.0,    0.0,    0.0,   0.0],
                                                    [0.0,   -self.Yvv,     0.0,    0.0,    0.0,   0.0],
                                                    [0.0,    0.0,    -self.Zww,    0.0,    0.0,   0.0],
                                                    [0.0,    0.0,     0.0,   -self.Kpp,    0.0,   0.0],
                                                    [0.0,    0.0,     0.0,    0.0,   -self.Mqq,   0.0],
                                                    [0.0,    0.0,     0.0,    0.0,    0.0,  -self.Nrr]])

        # Dynamics
        self.M = sym.Matrix(self.Mrb + self.Ma)
        self.C = self.coriolisMatrix(self.M, self.state)
        self.D = sym.Matrix(self.linear_damping + self.quadratic_damping);
        self.G = self.gravityMatrix(self.state)

        # Non-linear dynamics function of f (page 138 of Computer-Aided Control Systems Design, Chin 2013)
        f1 = sym.zeros(12, 12)
        f1[0:6, 6:12] = self.body2ned(self.state)
        f1[6:12, 6:12] = -self.M.LUsolve(self.C - self.D)
        f2 = sym.zeros(12, 1)
        f2[6:12, 0] = -self.M.LUsolve(self.G)

        self.f = f1@self.state + f2

        # Control
        # thrust allocation matrix
        self.thrust_allocation = np.zeros((thruster_count, 6))
        self.thrust_allocation[0:thruster_count, 0:3] = thruster_direction

        # maps XYZ torques
        for i in range(thruster_count):
            self.thrust_allocation[i, 3:6] = np.cross(thruster_position[i, 0:3], thruster_direction[i, 0:3])
        self.thrust_allocation = self.thrust_allocation.T

        # control input u
        eps = 1e-6
        self.u_control = sym.zeros(1, thruster_count)
        for i in range(thruster_count):
            self.u_control[i] = sym.Piecewise((self.du[i]**2, self.du[i] >= 0), (-self.du[i]**2, self.du[i] < 0))

        # generalized force tau
        self.tau = self.thrust_allocation @ self.u_control.T

        # control function g (page 138 of Computer-Aided Control Systems Design, Chin 2013)
        self.g = sym.zeros(12, 1)
        self.M_inv = self.M.inv()
        self.g[6:12, 0] = self.M_inv @ self.tau

        # state space
        # non-linear state space model F_dot
        self.F_dot = sym.zeros(12, 1)
        for i in range(len(self.f)):
            self.F_dot[i, 0] = self.f[i, 0] + self.g[i, 0]

        self.linearize()
        self.G = self.G.subs(parameter_map)
        self.G = sym.lambdify([self.roll, self.pitch, self.radius], self.G, modules="numpy")

    # return 3x3 anti-symmetric or skew-symmetric matrix
    def s(self, vec):
        return sym.Matrix([[0.0,        -vec[2],    vec[1]],
                           [vec[2],     0.0,        -vec[0]],
                           [-vec[1],    vec[0],     0.0]])

    # return 6x6 coriolis matrix (page 53 of Handbook of Marine Craft, 2011)
    def coriolisMatrix(self, M, state):
        v1 = state[6:9, 0]
        v2 = state[9:12, 0]

        s1 = self.s(M[0:3, 0:3] @ v1 + M[0:3, 3:6] @ v2)
        s2 = self.s(M[3:6, 0:3] @ v1 + M[3:6, 3:6] @ v2)
        C = sym.zeros(6, 6)
        C[0:3, 3:6] = -s1
        C[3:6, 0:3] = -s1
        C[3:6, 3:6] = -s2
        return C

    # return 6x1 gravity matrix (page 60 of Handbook of Marine Craft, 2011)
    def gravityMatrix(self, state):
        phi, theta, psi = state[3], state[4], state[5]
        # W: weight ... F_buoyancy: buoyancy force
        W = self.mass * self.gravity
        pi = 3.14159265359
        F_buoyancy = ((4/3)*pi*self.radius**3)*self.water_density*self.gravity

        # gravity center position in the robot fixed frame (gx,gy,gz) [m]
        gx, gy, gz = self.gravity_center[0], self.gravity_center[1], self.gravity_center[2]
        # buoyancy center position in the robot fixed frame (bx,by,bz) [m]
        bx, by, bz = self.buoyancy_center[0], self.buoyancy_center[1], self.buoyancy_center[2]

        G = sym.Matrix([[(W - F_buoyancy)*sin(theta)],
                        [-(W - F_buoyancy)*cos(theta)*sin(phi)],
                        [-(W - F_buoyancy)*cos(theta)*cos(phi)],
                        [-(gy*W - by*F_buoyancy)*cos(theta)*cos(phi) + (gz*W - bz*F_buoyancy)*cos(theta)*sin(phi)],
                        [(gz*W - bz*F_buoyancy)*sin(theta) + (gx*W - bx*F_buoyancy)*cos(theta)*cos(phi)],
                        [-(gx*W - bx*F_buoyancy)*cos(theta)*sin(phi) - (gy*W - by*F_buoyancy)*sin(theta)]])
        return G

    # return Transformation from BODY to NED coordinates (page 26 of Handbook of Marine Craft, 2011)
    def body2ned(self, state):
        phi, theta, psi = state[3], state[4], state[5]

        # the velocity is transformed from BODY to NED cooridnate system
        vel_NED = sym.Matrix([[cos(psi)*cos(theta), -sin(psi)*cos(phi)+cos(psi)*sin(theta)*sin(phi), sin(psi)*sin(phi)+cos(psi)*cos(phi)*sin(theta)],
                              [sin(psi)*cos(theta), cos(psi)*cos(phi)+sin(phi)*sin(theta)*sin(psi), -cos(psi)*sin(phi)+sin(theta)*sin(psi)*cos(theta)],
                              [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)]])

        # angular velocity is transformed from BODY to NED coordinate system
        angular_vel_NED = sym.Matrix([[1.0, sin(phi)*tan(theta), cos(phi)*tan(theta)],
                                      [0.0, cos(phi), -sin(phi)],
                                      [0.0, sin(phi)/cos(theta), cos(phi)/cos(theta)]])

        body2ned = sym.zeros(6, 6)
        body2ned[0:3, 0:3] = vel_NED
        body2ned[3:6, 3:6] = angular_vel_NED

        return body2ned

    # The system is linearized via the jacoboian
    def linearize(self):
        # substitude constant parameter
        self.state_dot = self.F_dot.subs(parameter_map)
        # print(parameter_map)
        # jacobian linearization with respect to state
        self.df_dstate_sym = self.state_dot.jacobian(self.state)
        # jacobian linearization with respect to control
        self.df_dcontrol_sym = self.state_dot.jacobian(self.du.T)


        self.df_dstate_funct = sym.lambdify([x, y, z, roll, pitch, yaw, u, v, w, p, q, r, radius], self.df_dstate_sym, modules="numpy")
        self.df_dcontrol_funct = sym.lambdify([self.du0, self.du1, self.du2, self.du3, self.du4, self.du5], self.df_dcontrol_sym, modules="numpy")
        self.df_dcontrol = self.df_dcontrol_funct(1, 1, 1, 1, 1, 1)
        # print(self.df_dcontrol.shape)


def main():
    state_space = StateSpace()


if __name__ == "__main__":
    main()
