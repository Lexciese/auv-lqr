import sympy as sym
import numpy as np

x, y, z, roll, pitch, yaw, u, v, w, p, q, r, du0, du1, du2, du3, du4, du5, radius = sym.symbols('x y z roll pitch yaw u v w p q r du0 du1 du2 du3 du4 du5 radius')
thruster_position = np.array([[-0.028, -0.296, 0.013],
                                   [-0.028,  0.296,  0.013],
                                   [-0.352,  0.000,  0.101],
                                   [0.352,  0.000,  0.101],
                                   [0.000,  0.005, -0.184],
                                   [0.000,  0.005,  0.184]])

thruster_direction = np.array([[1, 0, 0],
                                    [1, 0, 0],
                                    [0, 0, 1],
                                    [0, 0, 1],
                                    [0, 1, 0],
                                    [0, 1, 0]])

thruster_count = thruster_position.shape[0]


# Constant parameters (all in SI units)
# Gravity matrix parameters
displaced_water_volume = 0.035
water_density = 1000.0
gx = -0.0056
gy = 0.0002
gz = 0.0181
bx = -0.0055
by = 0.0007
bz = 0.0140
gravity = 9.81

# Mass matrix parameters
mass = 34
Ix = 1.44
Iy = 1.11
Iz = 1.79
Ixy = -0.01
Ixz = 0.066
Iyz = 0.006
mzg = mass*abs(gz)

# Added mass matrix parameters
added_mass = water_density*displaced_water_volume
mass_ratio = added_mass/mass
Xu_dot = mass_ratio*mass
Yv_dot = mass_ratio*mass
Zw_dot = mass_ratio*mass
Kp_dot = mass_ratio*Ix
Mq_dot = mass_ratio*Iy
Nr_dot = mass_ratio*Iz
Xq_dot = mass_ratio*mzg
Yp_dot = mass_ratio*mzg

# Damping matrix parameters
# Linear damping
Xu = 130.0;
Yv = 300.0;
Zw = 150.0;
Kp = 65.0;
Mq = 65.0;
Nr = 65.0;

# Quadratic Damping
Xuu = 155.0;
Yvv = 200.0;
Zww = 175.0;
Kpp = 95.0;
Mqq = 95.0;
Nrr = 95.0;


# if __name__ == "__main__":
#     config = Config()
