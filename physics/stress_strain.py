from __future__ import annotations
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import odeint
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from functools import partial

from tap import Tap


class Args(Tap):
    k1: float = 3
    k2: float = 3
    t1: float = 3
    t2: float = 3

    nadir: float = 1e-5
    apex: float = 5

    mass: float = 1
    gravity: float = 10


args = Args().parse_args()
k1_init = args.k1
k2_init = args.k2
t1_init = args.t1
t2_init = args.t2
mass = args.mass
gravity = args.gravity
nadir = args.nadir
apex = args.apex


class Params(object):
    def __init__(self, ks, ts):
        k1, k2 = ks
        t1, t2 = ts
        self.k1 = k1
        self.k2 = k2
        self.t1 = t1
        self.t2 = t2

    def update(self, ps: Params):
        self.k1 = ps.k1
        self.k2 = ps.k2
        self.t1 = ps.t1
        self.t2 = ps.t2


params = Params([k1_init, k2_init], [t1_init, t2_init])

def stress_func(params, x):
    def UStep(t):
        return 1 if t > 0 else 0
    k1 = params.k1
    k2 = params.k2
    t1 = params.t1
    t2 = params.t2
    gravity = 10

    x1 = k2 * x / (k1 + k2)
    x2 = k1 * x / (k1 + k2)
    neither_lock = (k1 * x1 + k2 * x2) * (UStep(t1 - x1) * UStep(t2 - x2))
    spring1_lock = (k1 * t1 + k2 * (x - t1)) * (UStep(x1 - t1) * UStep(t2 - x2))
    spring2_lock = (k2 * t2 + k1 * (x - t2)) * (UStep(t1 - x1) * UStep(x2 - t2))
    both_lock = gravity * (UStep(x1 - t1) * UStep(x2 - t2))
    ret = neither_lock + spring1_lock + spring2_lock + both_lock
    return ret


def spring_acc(disp):
    return gravity - stress_func(params, disp) / mass

def dposvel_dt(U, t):
    # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
    return [U[1], spring_acc(U[0])]


def forward_sim(ts):
    posvel0 = [0, 0]
    posvel = odeint(dposvel_dt, posvel0, ts)
    poss = posvel[:, 0]
    first_osilation = poss[0: int(num_increments * 0.3)]
    vels = posvel[:, 1]
    accs = np.array(list(map(spring_acc, poss)))
    tmaxp = ts[list(poss).index(max(first_osilation))]
    maxabsa = max(list(map(abs, accs)))
    print(f'At time {tmaxp}, the position is maximal, reaching {max(first_osilation)}; max abs acceleration is {maxabsa}')
    return np.array([poss, vels, accs])


# Initial x and y arrays
xs = np.linspace(0, 10, 1000)
y = np.array(list(map(partial(stress_func, params), xs)))
# y = np.sin(0.5 * x) * np.sin(x * np.random.randn(30))
# Spline interpolation
spline = UnivariateSpline(xs, y, s=6)
x_spline = np.linspace(0, 10, 1000)
y_spline = np.array(list(map(partial(stress_func, params), x_spline)))
# Plotting
fig = plt.figure()
plt.subplots_adjust(bottom=0.25)
(ax1, ax2) = fig.subplots(nrows=2, ncols=1)
stress_strain_plot = ax1.plot(xs, y)
stress_strain_plot, = ax1.plot(x_spline, y_spline, 'g')

num_increments = 200
ts = np.linspace(0, 10, num_increments)
init_sim = forward_sim(ts)
pos_plot, = ax2.plot(ts, init_sim[0], label='pos')
vel_plot, = ax2.plot(ts, init_sim[1], label='vel')
acc_plot, = ax2.plot(ts, init_sim[2], label='acc')
ax2.legend()


# Defining the Slider button
# xposition, yposition, width and height
k1_ax = plt.axes([0.25, 0.16, 0.65, 0.03])
k1_slider = Slider(k1_ax, 'k1', 0.1, 10, valinit=params.k1, valstep=0.1)

k2_ax = plt.axes([0.25, 0.11, 0.65, 0.03])
k2_slider = Slider(k2_ax, 'k2', 0.1, 10, valinit=params.k2, valstep=0.1)

t1_ax = plt.axes([0.25, 0.06, 0.65, 0.03])
t1_slider = Slider(t1_ax, 't1', 0.1, 10, valinit=params.t1, valstep=0.1)

t2_ax = plt.axes([0.25, 0.01, 0.65, 0.03])
t2_slider = Slider(t2_ax, 't2', 0.1, 10, valinit=params.t2, valstep=0.1)


# Updating the plot
def k1_update(val):
    new_param_update(Params([val, params.k2], [params.t1, params.t2]))
def k2_update(val):
    new_param_update(Params([params.k1, val], [params.t1, params.t2]))
def t1_update(val):
    new_param_update(Params([params.k1, params.k2], [val, params.t2]))
def t2_update(val):
    new_param_update(Params([params.k1, params.k2], [params.t1, val]))
def new_param_update(ps):
    params.update(ps)
    new_ys = np.array(list(map(partial(stress_func, params), x_spline)))
    # spline = UnivariateSpline(xs, new_ys, s=1)
    stress_strain_plot.set_ydata(new_ys)
    new_simdata = forward_sim(ts)
    pos_plot.set_ydata(new_simdata[0])
    vel_plot.set_ydata(new_simdata[1])
    acc_plot.set_ydata(new_simdata[2])
    # redrawing the figure
    ax1.relim()
    ax1.autoscale_view(True, True, True)
    ax2.relim()
    ax2.autoscale_view(True, True, True)
    fig.canvas.draw()


# Calling the function "update" when the value of the slider is changed
k1_slider.on_changed(k1_update)
k2_slider.on_changed(k2_update)
t1_slider.on_changed(t1_update)
t2_slider.on_changed(t2_update)
plt.show()



