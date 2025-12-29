import sympy as sym
import numpy as np

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


# # Constant parameters (all in SI units)

x, y, z, roll, pitch, yaw, u, v, w, p, q, r, du0, du1, du2, du3, du4, du5, du6, du7, radius = sym.symbols("x y z roll pitch yaw u v w p q r du0 du1 du2 du3 du4 du5 du6 du7 radius")

# Mrb matrix
mass, Ix, Iy, Iz, Ixy, Ixz, Iyz, mzg = sym.symbols("mass Ix Iy Iz Ixy Ixz Iyz mzg")

# Ma matrix
Xu_dot, Yv_dot, Zw_dot, Kp_dot, Mq_dot, Nr_dot, Xq_dot, Yp_dot = sym.symbols("Xu_dot Yv_dot Zw_dot Kp_dot Mq_dot Nr_dot Xq_dot Yp_dot")

# Damping matrices
Xu, Xuu, Yv, Yvv, Zw, Zww, Kp, Kpp, Mq, Mqq, Nr, Nrr = sym.symbols("Xu Xuu Yv Yvv Zw Zww Kp Kpp Mq Mqq Nr Nrr")

# G matrix
gx, gy, gz, bx, by, bz, gravity, radius, water_density = sym.symbols("x gy gz bx by bz gravity radius water_density")


parameter = {
    'displaced_water_volume': 0.035,
    'water_density': 1000.0,
    'gx': -0.0056,
    'gy': 0.0002,
    'gz': 0.0181,
    'bx': -0.0055,
    'by': 0.0007,
    'bz': 0.0140,
    'gravity': 9.81,
    'mass': 34.0,
    'Ix': 1.44,
    'Iy': 1.11,
    'Iz': 1.79,
    'Ixy': -0.01,
    'Ixz': 0.066,
    'Iyz': 0.006,
    'Xu': 130.0,
    'Yv': 300.0,
    'Zw': 150.0,
    'Kp': 65.0,
    'Mq': 65.0,
    'Nr': 65.0,
    'Xuu': 155.0,
    'Yvv': 200.0,
    'Zww': 175.0,
    'Kpp': 95.0,
    'Mqq': 95.0,
    'Nrr': 95.0,
}

added_mass = parameter['water_density']*parameter['displaced_water_volume']
mass_ratio = added_mass/parameter['mass']
parameter.update({
    'Xu_dot': mass_ratio*parameter['mass'],
    'Yv_dot': mass_ratio*parameter['mass'],
    'Zw_dot': mass_ratio*parameter['mass'],
    'Kp_dot': mass_ratio*parameter['Ix'],
    'Mq_dot': mass_ratio*parameter['Iy'],
    'Nr_dot': mass_ratio*parameter['Iz'],
    'Xq_dot': mass_ratio*parameter['mass']*abs(parameter['gz']),
    'Yp_dot': mass_ratio*parameter['mass']*abs(parameter['gz']),
    'mzg': parameter['mass']*abs(parameter['gz']),
})

parameter_map = {
    mass: parameter['mass'],
    Ix: parameter['Ix'],
    Iy: parameter['Iy'],
    Iz: parameter['Iz'],
    Ixy: parameter['Ixy'],
    Ixz: parameter['Ixz'],
    Iyz: parameter['Iyz'],
    mzg: parameter['mzg'],
    Xu_dot: parameter['Xu_dot'],
    Yv_dot: parameter['Yv_dot'],
    Zw_dot: parameter['Zw_dot'],
    Kp_dot: parameter['Kp_dot'],
    Mq_dot: parameter['Mq_dot'],
    Nr_dot: parameter['Nr_dot'],
    Xq_dot: parameter['Xq_dot'],
    Yp_dot: parameter['Yp_dot'],
    Xu: parameter['Xu'],
    Xuu: parameter['Xuu'],
    Yv: parameter['Yv'],
    Yvv: parameter['Yvv'],
    Zw: parameter['Zw'],
    Zww: parameter['Zww'],
    Kp: parameter['Kp'],
    Kpp: parameter['Kpp'],
    Mq: parameter['Mq'],
    Mqq: parameter['Mqq'],
    Nr: parameter['Nr'],
    Nrr: parameter['Nrr'],
    gx: parameter['gx'],
    gy: parameter['gy'],
    gz: parameter['gz'],
    bx: parameter['bx'],
    by: parameter['by'],
    bz: parameter['bz'],
    gravity: parameter['gravity'],
    water_density: parameter['water_density'],
}


# if __name__ == "__main__":
#     config = Config()
