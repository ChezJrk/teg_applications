import matplotlib.pyplot as plt
from tqdm import trange
from scipy.optimize import minimize
import numpy as np
from typing import List

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
from teg.derivs import FwdDeriv, RevDeriv
from physics.smooth import InvertSqrt, IsNotNan
from teg.math.smooth import Invert, Sqrt
from teg.eval import evaluate
from teg.passes.substitute import substitute
from teg.passes.simplify import simplify
from teg.passes.reduce import reduce_to_base


class Args:
    thresholds: List[Var] = [Var('threshold1', 2), Var('threshold2', 2)]
    scales: List[Var] = [Var('scale1', 2), Var('scale2', 1)]

    mass: Const = Const(1)
    gravity: Const = Const(10)

    nadir: Const = Const(1e-5)
    apex: Const = Const(5)

    num_samples: int = 10
    maxiter: int = 20

    backend: str = 'numpy'


def stress(strain: ITeg, args: Args) -> ITeg:
    """Stress curve given the strain for a string-bungee system.

    :strain: is downward displacement
    :threshold: is the string length s - the bungee rest length b
    :scale: is the elastic modulus of the bungee
    """
    scale1, scale2 = args.scales
    threshold1, threshold2 = args.thresholds
    g = args.gravity
    delta_x = strain
    delta_x1 = delta_x * scale2 / (scale1 + scale2)
    delta_x2 = delta_x * scale1 / (scale1 + scale2)

    delta_x1_raw = delta_x * scale2
    delta_x2_raw = delta_x * scale1
    scaled_thresh1 = (scale1 + scale2) * threshold1
    scaled_thresh2 = (scale1 + scale2) * threshold2

    neither_lock = IfElse((delta_x1_raw <= scaled_thresh1) & (delta_x2_raw <= scaled_thresh2), scale1 * delta_x1 + scale2 * delta_x2, 0)
    spring1_lock = IfElse((delta_x1_raw > scaled_thresh1) & (delta_x2_raw < scaled_thresh2), scale2 * delta_x, 0)
    spring2_lock = IfElse((delta_x1_raw < scaled_thresh1) & (delta_x2_raw > scaled_thresh2), scale1 * delta_x, 0)
    both_lock = IfElse((delta_x1_raw >= scaled_thresh1) & (delta_x2_raw >= scaled_thresh2), g, 0)

    e = IfElse(delta_x < 0, 0, neither_lock + spring1_lock + spring2_lock + both_lock)

    return e


def solve_for_time_given_position(args: Args):
    """The time at which the bungee system goes from the apex to the nadir,
    assuming that is initially at rest position (0,0) and velocity. """
    m, g = args.mass, args.gravity
    nadir, apex = args.nadir, args.apex

    disp = TegVar('disp')
    x_hat = TegVar('x_hat')

    # Solution to the second-order linear ODE
    inner_integrand = g - stress(disp, args) / m
    velocity = 2 * Teg(0, x_hat, inner_integrand, disp)
    expr = InvertSqrt(velocity)
    ode_solution_wrt_time = Teg(nadir, apex, expr, x_hat)

    return ode_solution_wrt_time, expr


def optimize(args: Args):
    """Optimizing both yield strength and scale for springiness. """
    g = args.gravity.value
    m = args.mass.value
    num_samples = args.num_samples
    loss_values = []
    scale_values = []
    threshold_values = []

    expr, invert_sqrt_vel = solve_for_time_given_position(args)
    expr = simplify(expr)
    # deriv = simplify(reduce_to_base(reverse_deriv(expr, output_list=[*args.scales, *args.thresholds], args={'ignore_deltas': True, 'ignore_bounds': True})[1]))
    deriv = simplify(reduce_to_base(reverse_deriv(expr, output_list=[*args.scales, *args.thresholds])[1]))
    deriv = simplify(RevDeriv(expr, Tup(Const(1))))

    def loss_and_grads(values):
        param_assigns = dict(zip(args.scales + args.thresholds, values))
        loss = evaluate(expr, param_assigns, num_samples=num_samples, backend=args.backend)
        vel_blows = invert_sqrt_vel.blow_up
        big_loss_const = 50
        loss = big_loss_const if vel_blows else loss
        invert_sqrt_vel.blow_up = False

        grads = evaluate(deriv, param_assigns, num_samples=num_samples, backend=args.backend)
        grads = (1, 1) if vel_blows else grads
        print(f'loss: {loss}')
        # print(f'scale: {scale_val}\n')

        loss_values.append(loss)
        scale_values.append([scale.value for scale in args.scales])
        threshold_values.append([threshold.value for threshold in args.thresholds])

        return loss, grads

    def max_acceleration_bounded(values):
        # max(|acc|) < g
        # since acc < 0, -acc >= 0
        # max(-acc) < g
        # max(-(stress(disp) - g)) < g
        # max(-stress(disp)) < 0
        # -max(-stress(disp)) > 0
        # min(stress) > 0

        # import ipdb; ipdb.set_trace()
        param_assigns = dict(zip(args.scales + args.thresholds, values))
        min_stress = min([evaluate(stress(Const(x), args), param_assigns, num_samples=num_samples, backend=args.backend)
                          for x in np.arange(args.nadir.value, args.apex.value, 0.1)])
        return min_stress

    def displacement_is_bounded(values):
        # total energy = elastic potential + gravitation potential + kinetic
        # gravitation potential > 0 at every position
        # total energy - max(elastic potential + kinetic) > 0

        # Assuming initially no elastic potential and at rest (no kinetic)
        # initial potential - max(work + kinetic)
        # initial potential - max_x(integral_x^apex force + kinetic)
        # (m=1)(g)(apex) - max_x((m=1)v(x) + 1/2 (m=1)v(x)^2) > 0

        def vel_squared(x_hat):
            disp = TegVar('disp')
            inner_integrand = stress(disp, args) / m - g
            velocity_squared = 2 * Teg(x_hat, args.apex, inner_integrand, disp)
            return velocity_squared

        def kinetic_elastic(x_hat):
            kinetic = m * vel_squared(x_hat) / 2
            elastic = vel_squared(x_hat) / 2
            return kinetic + elastic

        param_assigns = dict(zip(args.scales + args.thresholds, values))
        max_kinetic_elastic = max([evaluate(kinetic_elastic(x_hat), param_assigns, num_samples=num_samples, backend=args.backend)
                                   for x_hat in np.arange(args.nadir.value, args.apex.value, 0.1)])

        total_energy = m * g * args.apex.value
        return total_energy - max_kinetic_elastic

    ### loss function plotting
    # def compute_samples (sval, tval):
    #     print(f's: {sval} | t: {tval}')
    #     return loss_and_grads((sval, tval))

    # scales = np.arange(0, 10, step=1)
    # threshold_val = 9
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # plat = plt
    # vals = np.array([compute_samples(scale, threshold_val) for scale in scales])
    # axes[0].plot(scales, [v[0] for v in vals])
    # axes[1].plot(scales, [v[1] for v in vals])
    # plat.show()

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

    cons = [
        {'type': 'ineq', 'fun': max_acceleration_bounded},
        {'type': 'ineq', 'fun': displacement_is_bounded},
    ]

    options = {'maxiter': args.maxiter}
    print('Starting minimization')
    res = minimize(loss_and_grads, [var.value for var in (args.scales + args.thresholds)], constraints=cons, tol=1e-1, jac=True, options=options)
    print('The final parameter values are', res.x)
    print('Ending minimization')
    return loss_values, scale_values, threshold_values, res.x


if __name__ == "__main__":

    args = Args()
    scales_init, thresholds_init = [scale.value for scale in args.scales], [threshold.value for threshold in args.thresholds]
    loss_values, scale_values, threshold_values, final_param_vals = optimize(args)

    strains = np.arange(0, args.apex.value, step=0.01)
    param_assigns = dict(zip(args.scales + args.thresholds, scales_init + thresholds_init))
    stresses_before = [evaluate(stress(Const(strain), args), param_assigns, num_samples=args.num_samples, backend=args.backend)
                       for strain in strains]
    param_assigns = dict(zip(args.scales + args.thresholds, final_param_vals))
    stresses_after = [evaluate(stress(Const(strain), args), num_samples=args.num_samples, backend=args.backend)
                      for strain in strains]

    # fig, axes = plt.subplots(nrows=2, ncols=3)
    # plat = plt
    # plt = axes[0][0]

    plt.plot(strains, stresses_before, label='Before')
    plt.plot(strains, stresses_after, label='After')
    # plt.set(xlabel='Strain', ylabel='Stress')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.legend()
    plt.show()

    # plt = axes[0][1]
    # plt.plot(loss_values)
    # plt.set(xlabel='Iteration of Optimization', ylabel='Value of Loss')

    # plt = axes[1][0]
    # plt.plot(scale1_values)
    # plt.set(xlabel='Iteration of Optimization', ylabel='Scale1 of Stress')

    # plt = axes[1][1]
    # plt.plot(scale2_values)
    # plt.set(xlabel='Iteration of Optimization', ylabel='Scale2 of Stress')

    # plt = axes[0][2]
    # plt.plot(threshold1_values)
    # plt.set(xlabel='Iteration of Optimization', ylabel='Threshold1')

    # plt = axes[1][2]
    # plt.plot(threshold2_values)
    # plt.set(xlabel='Iteration of Optimization', ylabel='Threshold2')

    # plat.show()
