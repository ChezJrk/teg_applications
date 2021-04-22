from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.integrate import odeint
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm
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

    do_t1_solve: bool = True


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
teg_disp_constraint_grad_expr = simplify(FwdDeriv(teg_disp_constraint_expr, [(teg_params.t1, 1)]).deriv_expr)


def teg_disp_constraint(ps):
    param_assigns = dict(zip(teg_params.to_list(), ps.to_list()))
    loss = evaluate(teg_disp_constraint_expr, param_assigns, num_samples=num_samples, backend=args.backend)
    return loss

def teg_disp_constraint_grad(ps):
    param_assigns = dict(zip(teg_params.to_list(), ps.to_list()))
    loss = evaluate(teg_disp_constraint_grad_expr, param_assigns, num_samples=num_samples, backend=args.backend)
    return loss


def teg_acc():
    apex_acc = gravity - teg_stress(teg_params, args.apex)
    return apex_acc


teg_acc_constraint_expr = teg_acc()


def teg_acc_constraint(ps):
    param_assigns = dict(zip(teg_params.to_list(), ps.to_list()))
    loss = evaluate(teg_acc_constraint_expr, param_assigns, num_samples=num_samples, backend=args.backend)
    return loss

###


# Plotting
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')


num_k1s = 20
num_k2s = 20
k1s = np.linspace(0, 10, num_k1s)
k2s = np.linspace(0, 10, num_k2s)
K1s = np.array([k1s for _ in k2s])
K2s = np.array([k2s for _ in k1s]).transpose()


def eval_loss_surface():
    def solve_for_t1(k1, k2, t2):
        def t1_f(x):
            ps = Params([k1, k2], [x, t2])
            return teg_disp_constraint(ps)

        def t1_df(x):
            ps = Params([k1, k2], [x, t2])
            return teg_disp_constraint_grad(ps)

        def t1_loss(x):
            t1_val, = x
            return np.power(t1_f(t1_val) - args.apex, 2)

        def t1_jac(x):
            t1_val, = x
            return 2 * (t1_f(t1_val) - args.apex) * t1_df(t1_val)
        init_guess = np.array([2])
        # ret = minimize(t1_loss, init_guess, method='trust-constr', jac=t1_jac, bounds=[(0, 10)], options={'verbose': 0})
        ret = minimize(t1_loss, init_guess, jac=t1_jac)
        t1_val = np.clip(ret.x[0], 0, 10)
        # t1_val, = ret.x
        return t1_val

    vals = np.zeros((num_k1s, num_k2s))
    param_solves = [[None for _ in range(num_k2s)] for _ in range(num_k1s)]
    for i, k1 in enumerate(k1s):
        for j, k2 in enumerate(k2s):
            t1 = solve_for_t1(k1, k2, params.t2) if args.do_t1_solve else params.t1
            ps = Params([k1, k2], [t1, params.t2])
            val = teg_loss(ps)
            vals[i, j] = val
            param_solves[i][j] = ps
    return vals, param_solves


loss_vals, loss_param_solves = eval_loss_surface()
loss_plot_kwargs = {'rstride': 1, 'cstride': 1, 'alpha': 0.5}
loss_contour_kwargs = {'levels': [_ for _ in np.arange(0.5, 2, .1)], 'cmap': cm.get_cmap('magma'), 'linestyles': "solid"}
loss_plot = ax1.plot_surface(K1s, K2s, loss_vals, **loss_plot_kwargs)
ax1.contour(K1s, K2s, loss_vals, loss_vals, **loss_contour_kwargs)
ax1.set_xlabel('k1')
ax1.set_ylabel('k2')
ax1.set_zlabel('loss')


def eval_disp_surface(param_solves):
    vals = np.zeros((num_k1s, num_k2s))
    for i, k1 in enumerate(k1s):
        for j, k2 in enumerate(k2s):
            ps = param_solves[i][j]
            val = teg_disp_constraint(ps)
            vals[i, j] = val
    return vals


disp_vals = eval_disp_surface(loss_param_solves)
disp_plot_kwargs = {'rstride': 1, 'cstride': 1, 'alpha': 0.5}
disp_contour_kwargs = {'levels': [_ for _ in np.arange(0, 15, 5)], 'cmap': cm.get_cmap('magma'), 'linestyles': "solid"}
disp_plot = ax2.plot_surface(K1s, K2s, disp_vals, **disp_plot_kwargs)
ax2.contour(K1s, K2s, disp_vals, disp_vals, **disp_contour_kwargs)
ax2.set_xlabel('k1')
ax2.set_ylabel('k2')
ax2.set_zlabel('disp')


def eval_acc_surface(param_solves):
    vals = np.zeros((num_k1s, num_k2s))
    for i, k1 in enumerate(k1s):
        for j, k2 in enumerate(k2s):
            ps = param_solves[i][j]
            val = teg_acc_constraint(ps)
            vals[i, j] = val
    return vals


acc_vals = eval_acc_surface(loss_param_solves)
acc_plot_kwargs = {'rstride': 1, 'cstride': 1, 'alpha': 0.5}
acc_contour_kwargs = {'levels': [_ for _ in np.arange(-40, 20, 10)], 'cmap': cm.get_cmap('magma'), 'linestyles': "solid"}
acc_plot = ax3.plot_surface(K1s, K2s, acc_vals, **acc_plot_kwargs)
ax3.contour(K1s, K2s, acc_vals, acc_vals, **acc_contour_kwargs)
ax3.set_xlabel('k1')
ax3.set_ylabel('k2')
ax3.set_zlabel('apex acc')



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
    ax1.clear()
    new_loss_vals = eval_loss_surface()
    ax1.plot_surface(K1s, K2s, new_loss_vals, **loss_plot_kwargs)
    ax1.contour(K1s, K2s, loss_vals, loss_vals, **loss_contour_kwargs)

    ax2.plot_surface(K1s, K2s, disp_vals, **disp_plot_kwargs)
    ax2.contour(K1s, K2s, disp_vals, disp_vals, **disp_contour_kwargs)

    acc_plot = ax3.plot_surface(K1s, K2s, acc_vals, **acc_plot_kwargs)
    ax3.contour(K1s, K2s, acc_vals, acc_vals, **acc_contour_kwargs)

    # redrawing the figure
    ax1.relim()
    ax1.autoscale_view(True, True, True)
    ax2.relim()
    ax2.autoscale_view(True, True, True)
    ax3.relim()
    ax3.autoscale_view(True, True, True)
    fig.canvas.draw()


# Calling the function "update" when the value of the slider is changed
k1_slider.on_changed(k1_update)
k2_slider.on_changed(k2_update)
t1_slider.on_changed(t1_update)
t2_slider.on_changed(t2_update)
plt.show()



