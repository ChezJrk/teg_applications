import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import os
import pickle

from teg import (
    ITeg,
    Const,
    Var,
    IfElse,
    Teg,
    Tup,
    TegVar,
)
from teg.derivs.reverse_deriv import reverse_deriv
from teg.math.smooth import Sqrt, Sqr
from teg.eval import evaluate
from teg.passes.simplify import simplify
from teg.passes.reduce import reduce_to_base

from tap import Tap


class Args(Tap):
    k1: float = 1
    k2: float = 10
    l1: float = 2
    l2: float = 10

    nadir: float = 1e-5
    apex: float = 5.0
    plastic_weakening: float = 0.2

    mass: float = 1
    gravity: float = 10

    num_samples: int = 200
    maxiter: int = 1000
    tol: int = 1e-8

    finite_diff: bool = False
    ignore_deltas: bool = False

    backend: str = 'C'

    def process_args(self):
        self.thresholds = [Var('threshold1', self.l1), Var('threshold2', self.l2)]
        self.scales = [Var('scale1', self.k1), Var('scale2', self.k2)]


def stress(strain: ITeg, args: Args) -> ITeg:
    """Stress curve given the strain for a string-bungee system.

    :strain: is downward displacement
    :threshold: is the string length s - the bungee rest length b
    :scale: is the elastic modulus of the bungee

    stress = (k1 x1 + k2 x2) H(l1 - x1) H(l2 - x2) +
            (k1 l1 alpha + k2 (x - l1)) H(x1 - l1) H(l2 - x2) +
            (k2 l2 alpha + k1 (x - l1)) H(l1 - x1) H(x2 - l2) +
            g H(x1 - l1) H(x2 - l2)
    x1 = k2 x/(k1 + k2)
    x2 = k1 x/(k1 + k2)
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

    lock1 = scale1 * threshold1 * args.plastic_weakening + scale2 * (delta_x - threshold1)
    lock2 = scale2 * threshold2 * args.plastic_weakening + scale1 * (delta_x - threshold2)

    neither_lock = IfElse((delta_x1_raw <= scaled_thresh1) & (delta_x2_raw <= scaled_thresh2), scale1 * delta_x1 + scale2 * delta_x2, 0)
    spring1_lock = IfElse((delta_x1_raw > scaled_thresh1) & (delta_x2_raw < scaled_thresh2), lock1, 0)
    spring2_lock = IfElse((delta_x1_raw < scaled_thresh1) & (delta_x2_raw > scaled_thresh2), lock2, 0)
    both_lock = IfElse((delta_x1_raw >= scaled_thresh1) & (delta_x2_raw >= scaled_thresh2), g, 0)

    e = neither_lock + spring1_lock + spring2_lock + both_lock

    return e


def optimize(args: Args):
    """Optimizing both yield strength and scale for springiness. """
    g = args.gravity
    m = args.mass
    num_samples = args.num_samples
    nadir, apex = args.nadir, args.apex

    def generate_loss():
        def acc():
            return g - stress(apex, args) / m
        def t_from_pos():
            xhat = TegVar('xhat')
            disp = TegVar('disp')
            inner_integrand = g - stress(disp, args) / m
            hvs = Teg(0, xhat, inner_integrand, disp)
            inv_v = 1 / Sqrt(Sqrt(Sqr(hvs * 2)))
            t_of_apex = Teg(nadir, apex, inv_v, xhat)
            return t_of_apex

        out_list = [*args.scales, *args.thresholds]
        loss_expr = Sqr(acc() - 0.0) + t_from_pos()

        deriv_args = {'ignore_deltas': True} if args.ignore_deltas else {}

        loss_jac_list = reverse_deriv(loss_expr, Tup(Const(1)), output_list=out_list, args=deriv_args)[1]
        loss_jac_expr = simplify(reduce_to_base(loss_jac_list))

        def loss_f(values):
            param_assigns = dict(zip(args.scales + args.thresholds, values))
            loss = evaluate(loss_expr, param_assigns, num_samples=num_samples, backend=args.backend)
            return loss

        def loss_jac(values):
            param_assigns = dict(zip(args.scales + args.thresholds, values))
            grad = evaluate(loss_jac_expr, param_assigns, num_samples=num_samples, backend=args.backend)
            return grad

        return loss_f, loss_jac

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

        out_list = [*args.scales, *args.thresholds]
        deriv_args = {'ignore_deltas': True} if args.ignore_deltas else {}
        expr = half_vel_squared()
        expr_grad = reverse_deriv(expr, Tup(Const(1)), output_list=out_list, args=deriv_args)[1]
        expr_grad = simplify(reduce_to_base(expr_grad))

        def displacement_is_constrained(values):
            param_assigns = dict(zip(args.scales + args.thresholds, values))
            velocity_proxy = evaluate(expr, {**param_assigns}, num_samples=num_samples, backend=args.backend)
            return velocity_proxy

        def displacement_is_constrained_gradient(values):
            param_assigns = dict(zip(args.scales + args.thresholds, values))
            grad = evaluate(expr_grad, {**param_assigns}, num_samples=num_samples, backend=args.backend)
            return grad
        return displacement_is_constrained, displacement_is_constrained_gradient

    loss_f, loss_jac = generate_loss()
    disp_f, disp_jac = generate_displacement_is_constrained()
    cons = [
        {'type': 'eq', 'fun': disp_f} if args.finite_diff else {'type': 'eq', 'fun': disp_f, 'jac': disp_jac},
    ]

    print('Starting minimization')

    options = {'maxiter': args.maxiter, 'verbose': 2}
    init = [var.value for var in (args.scales + args.thresholds)]
    print(f'init loss    : {loss_f(init)}')
    method = 'trust-constr'
    bounds = ((0, 1000), (0, 1000), (0, 1000), (0, 1000))
    if args.finite_diff:
        res = minimize(loss_f, init, method=method, constraints=cons, bounds=bounds, options=options)
    else:
        res = minimize(loss_f, init, jac=loss_jac, method=method, constraints=cons, bounds=bounds, options=options)

    print('The final parameter values are', res.x)
    print('Command line args')
    k1, k2, l1, l2 = res.x
    print(f'--k1 {k1} --k2 {k2} --l1 {l1} --l2 {l2}')
    print(f'end loss     : {loss_f(res.x)}')
    print('Ending minimization')
    return res.x


def main():
    args = Args().parse_args()
    scales_init = [scale.value for scale in args.scales]
    thresholds_init = [threshold.value for threshold in args.thresholds]
    final_param_vals = optimize(args)
    num_samples = args.num_samples
    backend = args.backend

    strains = np.arange(0, args.apex+1, step=0.01)
    param_assigns = dict(zip(args.scales + args.thresholds, scales_init + thresholds_init))

    strain = Var('strain')
    stress_expr = stress(strain, args)
    stresses_before = [evaluate(stress_expr, {**param_assigns, strain: val}, num_samples=num_samples, backend=backend)
                       for val in strains]
    param_assigns = dict(zip(args.scales + args.thresholds, final_param_vals))
    stresses_after = [evaluate(stress_expr, {**param_assigns, strain: val}, num_samples=num_samples, backend=backend)
                      for val in strains]

    plt.plot(strains, stresses_before, label='Before')
    plt.plot(strains, stresses_after, label='After')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
