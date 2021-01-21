import matplotlib.pyplot as plt
from tqdm import trange
from scipy.optimize import minimize
import numpy as np

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
from teg.derivs import FwdDeriv, RevDeriv
from physics.smooth import InvertSqrt, IsNotNan
from teg.math.smooth import Invert, Sqrt
from teg.eval.numpy_eval import evaluate
from teg.passes.substitute import substitute
from teg.passes.simplify import simplify


def stress(strain: ITeg, threshold: Var, scale: Var) -> ITeg:
    """Stress curve given the strain for a string-bungee system. 

    :strain: is downward displacement
    :threshold: is the string length s - the bungee rest length b
    :scale: is the elastic modulus of the bungee
    """
    g = 10
    x = strain

    e = scale * x
    # e = IfElse(x < 0, 0, IfElse(x < threshold, scale * x, g))

    return e


def solve_for_time_given_position(nadir: Const, apex: Const, threshold: Var, scale: Var):
    """The time at which the bungee system goes from the apex to the nadir,
    assuming that is initially at rest position (0,0) and velocity. """
    disp = TegVar('disp')
    x_hat = TegVar('x_hat')
    g = 10

    # Solution to the second-order linear ODE
    inner_integrand = stress(disp, threshold, scale) - g
    velocity = -2 * Teg(0, x_hat, inner_integrand, disp)
    expr = InvertSqrt(velocity)
    ode_solution_wrt_time = Teg(nadir, apex, expr, x_hat)

    return ode_solution_wrt_time, expr


def optimize(nadir: Const, apex: Const, threshold: Var, scale: Var):
    """Optimizing both yield strength and scale for springiness. """
    g = 10
    # mass = 1
    num_samples = 100
    loss_values = []
    scale_values = []
    threshold_values = []

    expr, invert_sqrt_vel = solve_for_time_given_position(nadir, apex, threshold, scale)
    expr = simplify(expr)

    def loss_and_grads(args):
        scale_val, threshold_val = args
        scale.value = scale_val
        threshold.value = threshold_val

        loss = evaluate(expr, num_samples=num_samples, ignore_cache=True)
        vel_blows = invert_sqrt_vel.blow_up
        big_loss_const = 50
        loss = big_loss_const if vel_blows else loss
        invert_sqrt_vel.blow_up = False

        grads = evaluate(simplify(RevDeriv(expr, Tup(Const(1)))), num_samples=num_samples, ignore_cache=True)
        grads = (1, 1) if vel_blows else grads
        print(loss)
        loss_values.append(loss)
        scale_values.append(scale.value)
        threshold_values.append(threshold.value)
        return loss, grads

    def max_acceleration_bounded(args):
        # max(|acc|) < g
        # since acc < 0, -acc >= 0
        # max(-acc) < g
        # max(-(stress(disp) - g)) < g
        # max(-stress(disp)) < 0
        # -max(-stress(disp)) > 0
        # min(stress) > 0
        scale, threshold = args
        min_stress = min([evaluate(stress(Const(x), threshold, scale), num_samples=num_samples, ignore_cache=True)
                          for x in np.arange(nadir.value, apex.value, 0.1)])
        return min_stress

    def displacement_is_bounded(args):
        # total energy = elastic potential + gravitation potential + kinetic
        # gravitation potential > 0 at every position
        # total energy - max(elastic potential + kinetic) > 0

        # Assuming initially no elastic potential and at rest (no kinetic)
        # initial potential - max(work + kinetic)
        # initial potential - max_x(integral_x^apex force + kinetic)
        # (m=1)(g)(apex) - max_x((m=1)v(x) + 1/2 (m=1)v(x)^2) > 0
        scale, threshold = args

        def vel(x_hat):
            disp = TegVar('disp')
            inner_integrand = stress(disp, threshold, scale) - g
            velocity = 2 * Teg(x_hat, apex, inner_integrand, disp)
            return velocity

        def kinetic_elastic(x_hat):
            kinetic = vel(x_hat)**2 / 2
            elastic = vel(x_hat)
            return kinetic + elastic

        max_kinetic_elastic = max([evaluate(kinetic_elastic(x_hat), num_samples=num_samples, ignore_cache=True)
                                   for x_hat in np.arange(nadir.value, apex.value, 0.1)])

        total_energy = g * apex.value
        return total_energy - max_kinetic_elastic

    ### loss function plotting
    def compute_samples (sval, tval):
        print(f's: {sval} | t: {tval}')
        return loss_and_grads((sval, tval))

    scales = np.arange(0, 10, step=1)
    threshold_val = 9
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plat = plt
    vals = np.array([compute_samples(scale, threshold_val) for scale in scales])
    axes[0].plot(scales, [v[0] for v in vals])
    axes[1].plot(scales, [v[1] for v in vals])
    plat.show()

    # from matplotlib import cm
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax2 = fig.add_subplot(132, projection='3d')
    # # ax3 = fig.add_subplot(133, projection='3d')
    # # X, Y, Z = axes3d.get_test_data(0.05)
    #
    # def compute_samples (sval, tval):
    #     print(f's: {sval} | t: {tval}')
    #     return loss_and_grads((sval, tval))[0]
    #
    # scale_samples = np.arange(0, 10, .5)
    # threshold_samples = np.arange(0, 10, .5)
    # X = np.array([np.array([scale_val for _ in threshold_samples]) for scale_val in scale_samples])
    # Y = np.array([np.array([threshold_val for threshold_val in threshold_samples]) for _ in scale_samples])
    # Z = np.array([np.array([compute_samples(scale_val, threshold_val) for threshold_val in threshold_samples]) for scale_val in scale_samples])
    # levels = [_ for _ in np.arange(0, 10, 1)]
    #
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5)
    # ax.contour(X, Y, Z, levels=levels, cmap=cm.get_cmap('magma'), linestyles="solid")
    # ax.set_xlabel('scale')
    # ax.set_ylabel('threshold')
    # ax.set_zlabel('loss')
    #
    # plt.show()

    ### plotting end

    # cons = [
    #     {'type': 'ineq', 'fun': max_acceleration_bounded},
    #     {'type': 'ineq', 'fun': displacement_is_bounded},
    # ]
    #
    # minimize(loss_and_grads, [scale.value, threshold.value], constraints=cons, tol=1e-1, jac=True)

    return loss_values, scale_values, threshold_values


if __name__ == "__main__":
    # Parameters to optimize
    scale_init = 1
    threshold_init = 5
    scale = Var('scale', scale_init)
    threshold = Var('threshold', threshold_init)
    nadir = Const(1e-5)
    apex = Const(5)

    loss_values, scale_values, threshold_values = optimize(nadir, apex, threshold, scale)
    #
    # import numpy as np
    # strains = np.arange(0, apex.value, step=0.1)
    # stresses_before = [evaluate(stress(Const(strain), threshold_init, scale_init), ignore_cache=True)
    #                    for strain in strains]
    # stresses_after = [evaluate(stress(Const(strain), threshold_values[-1], scale_values[-1]), ignore_cache=True)
    #                   for strain in strains]
    # print(scale_values, threshold_values)
    #
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # plat = plt
    # plt = axes[0][0]
    #
    # plt.plot(strains, stresses_before, label='Before')
    # plt.plot(strains, stresses_after, label='After')
    # plt.set(xlabel='Strain', ylabel='Stress')
    # # plt.legend()
    #
    # plt = axes[0][1]
    # plt.plot(loss_values)
    # plt.set(xlabel='Iteration of Optimization', ylabel='Value of Loss')
    #
    # plt = axes[1][0]
    # plt.plot(scale_values)
    # plt.set(xlabel='Iteration of Optimization', ylabel='Scale of Stress')
    #
    # plt = axes[1][1]
    # plt.plot(threshold_values)
    # plt.set(xlabel='Iteration of Optimization', ylabel='Yield Strength Threshold')
    # plat.show()

b = 3
s = 5
h = 5
