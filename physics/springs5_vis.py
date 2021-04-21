from __future__ import annotations
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import odeint
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from functools import partial

from tap import Tap

from teg import (
    ITeg,
    Const,
    Var,
    IfElse,
    Teg,
    Tup,
    LetIn,
    TegVar,
)
from teg.derivs.reverse_deriv import reverse_deriv
from teg.derivs.fwd_deriv import fwd_deriv
from teg.derivs import FwdDeriv, RevDeriv
# from physics.smooth import InvertSqrt, IsNotNan
from teg.math.smooth import Invert, Sqrt
from teg.eval import evaluate
from teg.passes.substitute import substitute
from teg.passes.simplify import simplify
from teg.passes.reduce import reduce_to_base


class Args(Tap):
    k1: float = 3
    k2: float = 4
    t1: float = 3
    t2: float = 9

    nadir: float = 1e-5
    apex: float = 5

    mass: float = 1
    gravity: float = 10

    num_samples: int = 50
    backend: str = 'C'


args = Args().parse_args()
k1_init = args.k1
k2_init = args.k2
t1_init = args.t1
t2_init = args.t2
mass = args.mass
gravity = args.gravity
nadir = args.nadir
apex = args.apex
num_samples = args.num_samples


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

    def to_list(self):
        return [self.k1, self.k2, self.t1, self.t2]

params = Params([k1_init, k2_init], [t1_init, t2_init])


teg_params = Params([Var('k1', 0), Var('k2', 0)], [Var('t1', 0), Var('t2', 0)])


###

def teg_stress(ps:Params, x):
    k1, k2, t1, t2 = ps.to_list()
    g = args.gravity
    x1 = x * k2 / (k1 + k2)
    x2 = x * k1 / (k1 + k2)

    x1_raw = x * k2
    x2_raw = x * k1
    scaled_t1 = (k1 + k2) * t1
    scaled_t2 = (k1 + k2) * t2

    lock1 = k1 * t1 + k2 * (x - t1)
    lock2 = k2 * t2 + k1 * (x - t2)

    neither_lock = IfElse((x1_raw <= scaled_t1) & (x2_raw <= scaled_t2), k1 * x1 + k2 * x2, 0)
    spring1_lock = IfElse((x1_raw > scaled_t1) & (x2_raw <= scaled_t2), lock1, 0)
    spring2_lock = IfElse((x1_raw <= scaled_t1) & (x2_raw > scaled_t2), lock2, 0)
    both_lock = IfElse((x1_raw > scaled_t1) & (x2_raw > scaled_t2), g, 0)

    e = neither_lock + spring1_lock + spring2_lock + both_lock

    return e


def teg_stress_raw(ps:Params, x):
    k1, k2, t1, t2 = ps.to_list()
    g = args.gravity
    x1 = x * k2 / (k1 + k2)
    x2 = x * k1 / (k1 + k2)

    x1_raw = x * k2
    x2_raw = x * k1
    scaled_t1 = (k1 + k2) * t1
    scaled_t2 = (k1 + k2) * t2

    lock1 = k1 * t1 + k2 * (x - t1)
    lock2 = k2 * t2 + k1 * (x - t2)

    def ie(c, a, b):
        return a if c else b

    neither_lock = ie((x1_raw <= scaled_t1) & (x2_raw <= scaled_t2), k1 * x1 + k2 * x2, 0)
    spring1_lock = ie((x1_raw > scaled_t1) & (x2_raw <= scaled_t2), lock1, 0)
    spring2_lock = ie((x1_raw <= scaled_t1) & (x2_raw > scaled_t2), lock2, 0)
    both_lock = ie((x1_raw > scaled_t1) & (x2_raw > scaled_t2), g, 0)

    e = neither_lock + spring1_lock + spring2_lock + both_lock

    return e


def solve_for_time_given_position(args: Args):
    """The time at which the bungee system goes from the apex to the nadir,
    assuming that is initially at rest position (0,0) and velocity. """
    m, g = args.mass, args.gravity
    nadir, apex = args.nadir, args.apex

    disp = TegVar('disp')
    # x_hat = TegVar('x_hat')
    x_hat = Const(70, 'x_hat')

    # Solution to the second-order linear ODE
    inner_integrand = g - teg_stress(teg_params, disp) / m
    velocity = 2 * Teg(0, x_hat, inner_integrand, disp)
    # expr = InvertSqrt(velocity)
    expr = 1 / Sqrt(velocity)
    # ode_solution_wrt_time = Teg(nadir, apex, expr, x_hat)

    # return ode_solution_wrt_time, expr
    return expr

p_expr = solve_for_time_given_position(args)
p_expr = simplify(p_expr)
p_x_hat = TegVar('x_hat')
teg_loss_expr = Teg(nadir, apex, substitute(p_expr, Const(70, 'x_hat'), p_x_hat), p_x_hat)


def teg_loss(ps:Params):
    print(ps.to_list())
    param_assigns = dict(zip(teg_params.to_list(), ps.to_list()))
    loss = evaluate(teg_loss_expr, param_assigns, num_samples=num_samples, backend=args.backend)
    return loss


p_disp = TegVar('p_disp')
teg_stress_expr = teg_stress(teg_params, p_disp)


def teg_stress_func(x):
    param_assigns = dict(zip(teg_params.to_list() + [p_disp], params.to_list() + [x]))
    stress = evaluate(teg_stress_expr, param_assigns, num_samples=num_samples, backend=args.backend)
    return stress


def teg_disp():
    c_disp = TegVar('c_disp')
    inner_integrand = gravity - teg_stress(teg_params, c_disp) / mass
    half_velocity_squared = Teg(args.nadir, args.apex, inner_integrand, c_disp)
    return half_velocity_squared


teg_disp_constraint_expr = teg_disp()


def teg_disp_constraint(ps):
    param_assigns = dict(zip(teg_params.to_list(), ps.to_list()))
    loss = evaluate(teg_disp_constraint_expr, param_assigns, num_samples=num_samples, backend=args.backend)
    return loss

###

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


###

# Initial x and y arrays
xs = np.linspace(0, 10, 1000)
y = np.array(list(map(partial(stress_func, params), xs)))
# y = np.sin(0.5 * x) * np.sin(x * np.random.randn(30))
# Spline interpolation
spline = UnivariateSpline(xs, y, s=6)
x_spline = np.linspace(0, 10, 1000)
y_spline = np.array(list(map(partial(stress_func, params), x_spline)))
# Plotting
fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
((ax1, ax2, ax3), (ax4, ax5, ax6)) = fig.subplots(nrows=2, ncols=3)
stress_strain_plot_init, = ax1.plot(xs, y)
stress_strain_plot, = ax1.plot(x_spline, y_spline, 'g')

num_increments = 200
ts = np.linspace(0, 10, num_increments)
init_sim = forward_sim(ts)
pos_plot, = ax2.plot(ts, init_sim[0], label='pos')
vel_plot, = ax2.plot(ts, init_sim[1], label='vel')
acc_plot, = ax2.plot(ts, init_sim[2], label='acc')
ax2.legend()

teg_xs = np.linspace(0, 10, 1000)
teg_x_spline = np.linspace(0, 10, 1000)
teg_ys = np.array(list(map(partial(teg_stress_raw, params), teg_xs)))
teg_y_spline = np.array(list(map(partial(teg_stress_raw, params), teg_x_spline)))
teg_stress_strain_plot_init, = ax4.plot(xs, teg_ys)
teg_stress_strain_plot, = ax4.plot(x_spline, teg_y_spline, 'g')


def teg_loss_vary_t1(t1):
    pt1 = Params([params.k1, params.k2], [t1, params.t2])
    return teg_loss(pt1)
teg_num_increments = 20
teg_t1s = np.linspace(0, 10, teg_num_increments)
teg_losses = np.array(list(map(teg_loss_vary_t1, teg_t1s)))
teg_loss_plot, = ax5.plot(teg_t1s, teg_losses, label='loss')
# teg_vel_plot, = ax5.plot(teg_t1s, teg_init_sim[1], label='vel')
# teg_acc_plot, = ax5.plot(teg_t1s, teg_init_sim[2], label='acc')
ax5.legend()


def teg_disp_constraint_vary_t1(t1):
    pt1 = Params([params.k1, params.k2], [t1, params.t2])
    return teg_disp_constraint(pt1)
teg_num_increments_disps = 20
teg_t1_disps = np.linspace(0, 10, teg_num_increments_disps)
teg_disp_cons = np.array(list(map(teg_disp_constraint_vary_t1, teg_t1_disps)))
teg_disp_constraint_plot, = ax6.plot(teg_t1_disps, teg_disp_cons, label='disp')
ax6.legend()

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

    new_teg_ys = np.array(list(map(partial(teg_stress_raw, params), teg_x_spline)))
    teg_stress_strain_plot.set_ydata(new_teg_ys)

    new_teg_losses = np.array(list(map(teg_loss_vary_t1, teg_t1s)))
    teg_loss_plot.set_ydata(new_teg_losses)

    new_teg_disps = np.array(list(map(teg_disp_constraint_vary_t1, teg_t1s)))
    teg_disp_constraint_plot.set_ydata(new_teg_disps)

    # redrawing the figure
    ax1.relim()
    ax1.autoscale_view(True, True, True)
    ax2.relim()
    ax2.autoscale_view(True, True, True)
    ax3.relim()
    ax3.autoscale_view(True, True, True)
    ax4.relim()
    ax4.autoscale_view(True, True, True)
    ax5.relim()
    ax5.autoscale_view(True, True, True)
    ax6.relim()
    ax6.autoscale_view(True, True, True)
    fig.canvas.draw()


# Calling the function "update" when the value of the slider is changed
k1_slider.on_changed(k1_update)
k2_slider.on_changed(k2_update)
t1_slider.on_changed(t1_update)
t2_slider.on_changed(t2_update)
plt.show()



