from sympy import symbols, diff
from sympy import Heaviside as UnitStep
from sympy.solvers.ode import dsolve
from sympy import Function, Derivative
from sympy.abc import x as t

from scipy.integrate import odeint

import matplotlib.pyplot as plt
import numpy as np

m, g = 1, 10
s1, s2 = 1.5, 8
t1, t2 = 3.6, 20

x1 = t1/s2*(s1+s2)
x2 = t2/s1*(s1+s2)
hm = 2*s1*s2/(s1+s2)


def s(x):
    return (1-UnitStep(x-x1))*hm*x + (1-UnitStep(x-x2))*UnitStep(x-x1)*s2*x  + UnitStep(x-x2)*10


xs = np.arange(20, step=0.1)
stresses = [s(x).evalf() for x in xs]

# plt.plot(xs, stresses)
# plt.show()

pos = Function("pos")(t)
vel = diff(pos, t)
acc = diff(pos, t, t)


# m * X''[t] ==m * g - s[X[t]], X[0]==0, X'[0] ==0} ,X, {t, 0, 10}
sol = dsolve(m * acc - m * g + s(pos), pos, ics={pos.subs(t, 0): 0, vel.subs(t, 0): 0})
print(sol)

