import numpy as np
import control as ct

class LQR():
    def __init__(self, state, target_state, df_dstate, B, G, thrust_allocation, Q_value, R_value):
        self.numthrusters = B.shape[1]
        # current_state
        self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.u, self.v, self.w, self.p, self.q, self.r = state
        self.state = state
        self.target_state = target_state
        self.df_dstate = df_dstate
        self.B = B
        self.G = G
        self.thrust_allocation = thrust_allocation

        self.du = None

        # tune parameter
        self.vehicle_radius = 0.2
        self.Q = np.diag(Q_value)
        self.R = R_value * np.eye(self.numthrusters)

        # thruster conversion factor
        self.force2thrusteffort_forward = 0.1
        self.force2thrusteffort_backward = 1.0 * self.force2thrusteffort_forward

        # if the sub moves out of the water, the flotability decrease
        if self.z < 0.0:
            self.radius = self.vehicle_radius - abs(self.z)
            if self.radius < 0.0:
                self.radius = 0.0
        else:
            self.radius = self.vehicle_radius

    # the lqr control loop
    def run(self, state):
        x, y, z, roll, pitch, yaw, u, v, w, p, q, r = state
        self.gravity_effects = self.G(roll, pitch, self.radius)

        # calculate A matrix
        self.A = self.df_dstate(x, y, z, roll, pitch, yaw, u, v, w, p, q, r, self.radius)

        # calculate lqr error
        self.lqr_error = state - self.target_state.T

        # Minimum distances between angles are used in  the error
        # https://stackoverflow.com/a/2007279
        for i in range(3):
            self.lqr_error[3 + i] = np.arctan2(
                np.sin(state[3 + i] - self.target_state[3 + i]),
                np.cos(state[3 + i] - self.target_state[3 + i])
            )
        # print(f"state: {self.state[5]}, target: {self.target_state[5]}, yaw error: {self.lqr_error[5]}")

        try:
            self.K, _, _ = ct.lqr(self.A, self.B, self.Q, self.R)
            # control thrust du in Newtons
            self.du = -self.K @ self.lqr_error
            # convert thrust-force(N) to thrust-effort(%)
            force2thrusteffort = np.zeros(self.numthrusters)
            i1 = np.where(self.du >= 0.0)[0]
            i2 = np.where(self.du <= 0.0)[0]
            force2thrusteffort[i1] = self.force2thrusteffort_forward
            force2thrusteffort[i2] = self.force2thrusteffort_backward
            self.du = np.diag(force2thrusteffort) @ self.du

            # modify "du" to counteract the gravity and the buoyancy effect
            force2thrusteffort = np.ones(self.numthrusters) * self.force2thrusteffort_forward
            du_gravity_effects = np.linalg.lstsq(self.thrust_allocation, self.gravity_effects, rcond=None)[0].flatten() # maps the force to thrust (N)
            du_gravity_effects = force2thrusteffort * du_gravity_effects # convert thrust-force to thrust-effort
            # self.du = self.du - du_gravity_effects


        except Exception as e:
            print(f"LQR calculation have error: {e}")

        # print(f"state: {state}")
        # print(self.A.shape, self.B.shape, self.du.shape)
        # print(f"A: {self.A}")
        # print(f"B: {self.B}")
        # print(f"DU: {self.du}")
        # print(f"K: {self.K}")
        # print(f"self.thrust_allocation: {self.thrust_allocation}")
        return self.A, self.B, self.du

