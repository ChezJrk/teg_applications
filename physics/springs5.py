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
    s1: float = 1
    s2: float = 10
    t1: float = 2
    t2: float = 10

    nadir: float = 1e-5
    apex: float = 5.0
    plastic_weakening: float = 0.2

    mass: float = 1
    gravity: float = 10

    num_samples: int = 200
    maxiter: int = 1000
    tol: int = 1e-8
    second_order: bool = False
    no_derivs: bool = False

    ignore_deltas: bool = False
    backend: str = 'C'
    deriv_cache: str = '/Users/undefined/Documents/GitHub/teg_applications/physics/springs_cached_derivs'
    # deriv_cache: str = './physics/springs_cached_derivs'

    def process_args(self):
        self.thresholds = [Var('threshold1', self.t1), Var('threshold2', self.t2)]
        self.scales = [Var('scale1', self.s1), Var('scale2', self.s2)]


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

        ignore_deltas = 'no_delta_' if args.ignore_deltas else ''
        deriv_path = os.path.join(args.deriv_cache, f'{ignore_deltas}deriv.pkl')
        second_deriv_path = os.path.join(args.deriv_cache, f'{ignore_deltas}second_deriv.pkl')
        # if not os.path.isfile(second_deriv_path if args.second_order else deriv_path):
        if True:
            import sys
            sys.setrecursionlimit(10000)

            print('Computing the first derivative')
            deriv_args = {'ignore_deltas': True} if args.ignore_deltas else {}

            loss_jac_list = reverse_deriv(loss_expr, Tup(Const(1)), output_list=out_list, args=deriv_args)[1]
            loss_jac_expr = simplify(reduce_to_base(loss_jac_list))

            pickle.dump(loss_jac_expr, open(deriv_path, "wb"))

            loss_hess_expr_list = []
            if args.second_order:
                print('Computing the second derivative')

                for i, eltwise_deriv in enumerate(loss_jac_list):
                    print(f'Iteration {i}: second derivative.')

                    eltwise_deriv = simplify(reduce_to_base(eltwise_deriv))
                    print('Computing reverse derivative')
                    sndd = reverse_deriv(eltwise_deriv, output_list=out_list)[1]
                    print('Reducing to base')
                    reduced_sndd = reduce_to_base(sndd, timing=True)
                    print('Simplifying')
                    res = simplify(reduced_sndd)
                    second_deriv_i = res
                    loss_hess_expr_list.append(second_deriv_i)

                pickle.dump(loss_hess_expr_list, open(second_deriv_path, "wb"))

        else:
            loss_jac_expr = pickle.load(open(deriv_path, "rb"))
            loss_hess_expr_list = pickle.load(open(second_deriv_path, "rb")) if args.second_order else []

        def loss_f(values):
            param_assigns = dict(zip(args.scales + args.thresholds, values))
            loss = evaluate(loss_expr, param_assigns, num_samples=num_samples, backend=args.backend)
            return loss

        def loss_jac(values):
            param_assigns = dict(zip(args.scales + args.thresholds, values))
            grad = evaluate(loss_jac_expr, param_assigns, num_samples=num_samples, backend=args.backend)
            return grad

        def loss_hess(values):
            param_assigns = dict(zip(args.scales + args.thresholds, values))
            hess = [evaluate(eltwise, param_assigns, num_samples=num_samples, backend=args.backend)
                  for eltwise in loss_hess_expr_list]
            return hess
        return loss_f, loss_jac, loss_hess

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

    loss_f, loss_jac, loss_hess = generate_loss()
    disp_f, disp_jac = generate_displacement_is_constrained()
    cons = [
        {'type': 'eq', 'fun': disp_f} if args.no_derivs else {'type': 'eq', 'fun': disp_f, 'jac': disp_jac},
    ]

    print('Starting minimization')

    options = {'maxiter': args.maxiter, 'verbose': 2}
    init_guess = [var.value for var in (args.scales + args.thresholds)]
    print(f'init loss    : {loss_f(init_guess)}')
    opt_bounds = ((0, 1000), (0, 1000), (0, 1000), (0, 1000))
    if args.no_derivs:
        res = minimize(loss_f, init_guess, method='trust-constr', constraints=cons, bounds=opt_bounds, options=options)
    elif args.second_order:
        res = minimize(loss_f, init_guess, jac=loss_jac, hess=loss_hess, method='trust-constr', constraints=cons, bounds=opt_bounds, options=options)
    else:
        res = minimize(loss_f, init_guess, jac=loss_jac, method='trust-constr', constraints=cons, bounds=opt_bounds, options=options)

    print('The final parameter values are', res.x)
    print('Command line args')
    s1, s2, t1, t2 = res.x
    print(f'--s1 {s1} --s2 {s2} --t1 {t1} --t2 {t2}')
    print(f'end loss     : {loss_f(res.x)}')
    # print(res)
    print('Ending minimization')
    return res.x


def main():
    args = Args().parse_args()
    scales_init, thresholds_init = [scale.value for scale in args.scales], [threshold.value for threshold in args.thresholds]
    final_param_vals = optimize(args)

    strains = np.arange(0, args.apex+1, step=0.01)
    param_assigns = dict(zip(args.scales + args.thresholds, scales_init + thresholds_init))

    strain = Var('strain')
    stress_expr = stress(strain, args)
    stresses_before = [evaluate(stress_expr, {**param_assigns, strain: val}, num_samples=args.num_samples, backend=args.backend)
                       for val in strains]
    param_assigns = dict(zip(args.scales + args.thresholds, final_param_vals))
    stresses_after = [evaluate(stress_expr, {**param_assigns, strain: val}, num_samples=args.num_samples, backend=args.backend)
                      for val in strains]

    plt.plot(strains, stresses_before, label='Before')
    plt.plot(strains, stresses_after, label='After')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()