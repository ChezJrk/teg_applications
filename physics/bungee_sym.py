from sympy import Heaviside as UnitStep

from scipy.integrate import odeint
from scipy import integrate

import matplotlib.pyplot as plt
import numpy as np

nadir = 1e-5
apex = 5

m, g = 1, 10
s1, s2 = 1.5, 8
t1, t2 = 3.6, 20

x1 = t1 / s2 * (s1 + s2)
x2 = t2 / s1 * (s1 + s2)
hm = 2 * s1 * s2 / (s1 + s2)


def s(x):
    return (1 - UnitStep(x - x1)) * hm * x + (1 - UnitStep(x - x2)) * UnitStep(x - x1) * s2 * x + UnitStep(x - x2) * 10


def spring_acc(disp):
    return g - s(disp) / m


def dposvel_dt(U, t):
    # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
    return [U[1], spring_acc(U[0])]


def ode_solution_wrt_time_np():
    def f(x_hat):
        velocity = 2 * integrate.quad(spring_acc, 0, x_hat)[0]
        expr = 1 / np.sqrt(velocity)
        return expr

    res = integrate.quad(f, nadir, apex)[0]
    return res


def f(x_hat):
    print(x_hat)
    velocity = 2 * integrate.quad(spring_acc, 0, x_hat)[0]
    expr = 1 / np.sqrt(velocity)
    return expr


poss_temp = np.linspace(nadir, apex, 100)[1:-1]
fs = list(map(f, poss_temp))

posvel0 = [0, 0]
xs = np.linspace(0, 10, 200)
posvel = odeint(dposvel_dt, posvel0, xs)
poss = posvel[:, 0]
vels = posvel[:, 1]
accs = list(map(spring_acc, poss))

fig, axes = plt.subplots(nrows=1, ncols=2)
plat = plt

plt = axes[0]
plt.set(xlabel='t', ylabel='poss')
# plt.title("Harmonic oscillator")
plt.plot(xs, poss, label='pos')
plt.plot(xs, vels, label='vel')
plt.plot(xs, accs, label='acc')
plt.legend()

plt = axes[1]
# plt.title("hello oscillator")
plt.plot(poss_temp, fs)

plat.show()
