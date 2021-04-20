import matplotlib.pyplot as plt
from tqdm import trange
from scipy.optimize import minimize
import numpy as np
from typing import List
import os
import pickle

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

from tap import Tap


class Args(Tap):
    s1: float = 3
    s2: float = 3
    t1: float = 3
    t2: float = 3

    nadir: float = 1e-5
    apex: float = 5

    mass: float = 1
    gravity: float = 10

    num_samples: int = 50
    maxiter: int = 40
    tol: int = 1e-8

    ignore_deltas: bool = False
    backend: str = 'C'
    deriv_cache: str = './physics/springs_cached_derivs'

    def process_args(self):
        self.thresholds = [Var('threshold1', self.t1), Var('threshold2', self.t2)]
        self.scales = [Var('scale1', self.s1), Var('scale2', self.s2)]


def stress(strain: ITeg, args: Args) -> ITeg:
    """Stress curve given the strain for a string-bungee system.

    :strain: is downward displacement
    :threshold: is the string length s - the bungee rest length b
    :scale: is the elastic modulus of the bungee

    (k1x1 + k2x2) H(l1 - x1)H(l2 - x2) + (k1(x - l1) + k2 l1) H(l1 - x1)H(x2 - l2) + (k2(x - l2) + k1 l2) H(x1 - l1) H(l2 - x2) + g H(x1 - l1) H(x2 - l2)
    x1 = k2 x/(k1 + k2), x2 = k1 x/(k1 + k2)
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

    lock1 = scale1 * threshold1 + scale2 * (delta_x - threshold1)
    lock2 = scale2 * threshold2 + scale1 * (delta_x - threshold2)

    neither_lock = IfElse((delta_x1_raw <= scaled_thresh1) & (delta_x2_raw <= scaled_thresh2), scale1 * delta_x1 + scale2 * delta_x2, 0)
    spring1_lock = IfElse((delta_x1_raw > scaled_thresh1) & (delta_x2_raw < scaled_thresh2), lock1, 0)
    spring2_lock = IfElse((delta_x1_raw < scaled_thresh1) & (delta_x2_raw > scaled_thresh2), lock2, 0)
    both_lock = IfElse((delta_x1_raw >= scaled_thresh1) & (delta_x2_raw >= scaled_thresh2), g, 0)

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
    inner_integrand = g - stress(disp, args) / m
    velocity = 2 * Teg(0, x_hat, inner_integrand, disp)
    # expr = InvertSqrt(velocity)
    expr = 1 / Sqrt(velocity)
    # ode_solution_wrt_time = Teg(nadir, apex, expr, x_hat)

    # return ode_solution_wrt_time, expr
    return expr


def optimize(args: Args):
    """Optimizing both yield strength and scale for springiness. """
    g = args.gravity
    m = args.mass
    num_samples = args.num_samples
    nadir, apex = args.nadir, args.apex
    loss_values = []
    scale_values = []
    threshold_values = []

    # expr, invert_sqrt_vel = solve_for_time_given_position(args)
    expr = solve_for_time_given_position(args)
    expr = simplify(expr)
    x_hat = TegVar('x_hat')
    loss_expr = Teg(nadir, apex, substitute(expr, Const(70, 'x_hat'), x_hat), x_hat)
    # deriv = simplify(reduce_to_base(reverse_deriv(expr, output_list=[*args.scales, *args.thresholds], args={'ignore_deltas': True, 'ignore_bounds': True})[1]))

    ignore_deltas = 'no_delta_' if args.ignore_deltas else ''
    deriv_path = os.path.join(args.deriv_cache, f'{ignore_deltas}deriv.pkl')
    second_deriv_path = os.path.join(args.deriv_cache, f'{ignore_deltas}second_deriv.pkl')
    if not os.path.isfile(second_deriv_path):
        print('Computing the first derivative')
        out_list = [*args.scales, *args.thresholds]
        silly_deriv = reverse_deriv(expr, output_list=out_list)[1]
        deriv = simplify(reduce_to_base(silly_deriv))
        deriv = Teg(nadir, apex, substitute(deriv, Const(70, 'x_hat'), x_hat), x_hat)

        print('Computing the second derivative')

        second_deriv = []
        for i, eltwise_deriv in enumerate(silly_deriv):
            print(f'Iteration {i}: second derivative.')

            eltwise_deriv = simplify(reduce_to_base(eltwise_deriv))
            print('Computing reverse derivative')
            sndd = reverse_deriv(eltwise_deriv, output_list=out_list)[1]
            print('Reducing to base')
            reduced_sndd = reduce_to_base(sndd, timing=True)
            print('Simplifying')
            res = simplify(reduced_sndd)
            second_deriv_i = substitute(res, Const(70, 'x_hat'), x_hat)
            second_deriv_i = Teg(nadir, apex, second_deriv_i, x_hat)
            second_deriv.append(second_deriv_i)

        pickle.dump(deriv, open(deriv_path, "wb"))
        pickle.dump(second_deriv, open(second_deriv_path, "wb"))

    else:
        deriv = pickle.load(open(deriv_path, "rb"))
        second_deriv = pickle.load(open(second_deriv_path, "rb"))

    def loss(values):
        param_assigns = dict(zip(args.scales + args.thresholds, values))
        s1, s2, t1, t2 = values
        print(f'--s1 {s1} --s2 {s2} --t1 {t1} --t2 {t2}')
        loss = evaluate(loss_expr, param_assigns, num_samples=num_samples, backend=args.backend)
        print(f'loss: {loss}')

        loss_values.append(loss)
        scale_values.append([scale.value for scale in args.scales])
        threshold_values.append([threshold.value for threshold in args.thresholds])

        return loss

    def jac(values):
        param_assigns = dict(zip(args.scales + args.thresholds, values))
        grads = evaluate(deriv, param_assigns, num_samples=num_samples, backend=args.backend)
        print(f'grad: {grads}')
        return grads

    def hess(values):
        param_assigns = dict(zip(args.scales + args.thresholds, values))
        hesses = [evaluate(eltwise, param_assigns, num_samples=num_samples, backend=args.backend)
                  for eltwise in second_deriv]
        print(f'hess: {hesses}')
        return hesses

    def generate_max_acceleration_is_bounded():
        # max(|acc|) < 2g
        # since acc < 0, -acc >= 0
        # max(-acc) < 2g
        # max(-(stress(disp) - g)) < 2g
        # max(-stress(disp)) < g
        # -max(-stress(disp)) + g > 0
        # min(stress) + g > 0
        x = Var('x')
        expr_to_min = stress(x, args) + g

        def max_acceleration_is_bounded(values):
            param_assigns = dict(zip(args.scales + args.thresholds, values))
            min_stress = min([evaluate(expr_to_min, {**param_assigns, x: val}, num_samples=num_samples, backend=args.backend)
                              for val in np.arange(args.nadir, args.apex, 0.1)])
            return min_stress

        return max_acceleration_is_bounded

    def generate_displacement_is_constrained():
        # x'' = f(x), x(0) = 0, x'(0) = 0
        # x' = ±sqrt(2(F(x) - F(0)))
        # x' = 0 when F(x) - F(0) = 0
        # max displacement occurs at position s when int_0^s f(z) dz = 0

        def half_vel_squared():
            disp = TegVar('disp')
            inner_integrand = g - stress(disp, args) / m
            half_velocity_squared = Teg(args.nadir, args.apex, inner_integrand, disp)
            return half_velocity_squared

        expr = half_vel_squared()
        def displacement_is_constrained(values):
            param_assigns = dict(zip(args.scales + args.thresholds, values))
            velocity_proxy = evaluate(expr, {**param_assigns}, num_samples=num_samples, backend=args.backend)
            return velocity_proxy
        return displacement_is_constrained

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
        {'type': 'ineq', 'fun': generate_max_acceleration_is_bounded()},
        {'type': 'eq', 'fun': generate_displacement_is_constrained()},
    ]

    options = {'maxiter': args.maxiter, 'verbose': 2}
    print('Starting minimization')

    init_guess = [var.value for var in (args.scales + args.thresholds)]
    res = minimize(loss, init_guess, jac=jac, hess=hess, method='trust-constr', constraints=cons, bounds=((0, 10), (0, 10), (0, 10), (0, 10)), options=options)

    # res = minimize(loss_and_grads, [var.value for var in (args.scales + args.thresholds)], constraints=cons, tol=args.tol, jac=True, options=options, bounds=((0, 10), (0, 10), (0, 10), (0, 10)))
    print('The final parameter values are', res.x)
    print('Command line args')
    s1, s2, t1, t2 = res.x
    print(f'--s1 {s1} --s2 {s2} --t1 {t1} --t2 {t2}')
    print('Ending minimization')
    return loss_values, scale_values, threshold_values, res.x


if __name__ == "__main__":

    args = Args().parse_args()
    scales_init, thresholds_init = [scale.value for scale in args.scales], [threshold.value for threshold in args.thresholds]
    loss_values, scale_values, threshold_values, final_param_vals = optimize(args)

    strains = np.arange(0, args.apex, step=0.01)
    param_assigns = dict(zip(args.scales + args.thresholds, scales_init + thresholds_init))

    strain = Var('strain')
    stress_expr = stress(strain, args)
    stresses_before = [evaluate(stress_expr, {**param_assigns, strain: val}, num_samples=args.num_samples, backend=args.backend)
                       for val in strains]
    param_assigns = dict(zip(args.scales + args.thresholds, final_param_vals))
    stresses_after = [evaluate(stress_expr, {**param_assigns, strain: val}, num_samples=args.num_samples, backend=args.backend)
                      for val in strains]

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
