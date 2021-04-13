from sympy import Heaviside as UnitStep

from scipy.integrate import odeint
from scipy import integrate

import matplotlib.pyplot as plt
import numpy as np

from tap import Tap


class Args(Tap):
    s1: float = 1.5
    s2: float = 8
    t1: float = 3.6
    t2: float = 20
    # s1: float = 1.5
    # s2: float = 8
    # t1: float = 3.6
    # t2: float = 20

    nadir: float = 1e-5
    apex: float = 5

    mass: float = 1
    gravity: float = 10


args = Args().parse_args()
s1 = args.s1
s2 = args.s2
t1 = args.t1
t2 = args.t2
mass = args.mass
gravity = args.gravity
nadir = args.nadir
apex = args.apex


def s(x):
    x1 = s1 * x / (s1 + s2)
    x2 = s2 * x / (s1 + s2)
    neither_lock = s1 * x1 + s2 * x2 * UnitStep(t1 - x1) * UnitStep(t2 - x2)
    spring1_lock = s1 * (x - t1) + s2 * t1 * UnitStep(t1 - x1) * UnitStep(x2 - t2)
    spring2_lock = s2 * (x - t2) + s1 * t2 * UnitStep(x1 - t1) * UnitStep(t2 - x2)
    both_lock = gravity * UnitStep(x1 - t1) * UnitStep(x2 - t2)
    return neither_lock + spring1_lock + spring2_lock + both_lock


def spring_acc(disp):
    return gravity - s(disp) / mass


def dposvel_dt(U, t):
    # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
    return [U[1], spring_acc(U[0])]


def ode_solution_wrt_time_np():
    def f(x_hat):
        velocity_squared = 2 * integrate.quad(spring_acc, 0, x_hat)[0]
        expr = 1 / np.sqrt(velocity_squared)
        return expr

    res = integrate.quad(f, nadir, apex)[0]
    return res


def f(x_hat):
    velocity_squared = 2 * integrate.quad(spring_acc, 0, x_hat)[0]
    expr = 1 / np.sqrt(velocity_squared)
    return expr


poss_temp = np.linspace(nadir, apex, 100)[1:-1]
fs = list(map(f, poss_temp))

posvel0 = [0, 0]
num_increments = 200
ts = np.linspace(0, 10, num_increments)
posvel = odeint(dposvel_dt, posvel0, ts)
poss = posvel[:, 0]
first_osilation = poss[0: int(num_increments * 0.3)]
print(f'At time {ts[list(poss).index(max(first_osilation))]}, the position is maximal, reaching {max(first_osilation)}')
vels = posvel[:, 1]
accs = list(map(spring_acc, poss))

fig, axes = plt.subplots(nrows=1, ncols=1)
plat = plt

plt = axes
plt.set(xlabel='t', ylabel='poss')
# plt.title("Harmonic oscillator")
plt.plot(ts, poss, label='pos')
plt.plot(ts, vels, label='vel')
plt.plot(ts, accs, label='acc')
plt.legend()

# plt = axes[1]
# plt.plot(poss_temp, fs)

plat.show()
