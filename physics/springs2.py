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


def stress(strain: ITeg, threshold1: Var, threshold2: Var, scale1: Var, scale2: Var) -> ITeg:
    """Stress curve given the strain for a string-bungee system.

    :strain: is downward displacement
    :threshold: is the string length s - the bungee rest length b
    :scale: is the elastic modulus of the bungee
    """
    g = 10
    delta_x = strain
    delta_x1 = delta_x * scale2 / (scale1 + scale2)
    delta_x2 = delta_x * scale1 / (scale1 + scale2)

    delta_x1_raw = delta_x * scale2
    delta_x2_raw = delta_x * scale1
    scaled_thresh1 = (scale1 + scale2) * threshold1
    scaled_thresh2 = (scale1 + scale2) * threshold2

    neither_lock = IfElse((delta_x1_raw < scaled_thresh1) & (delta_x2_raw < scaled_thresh2), scale1 * delta_x1 + scale2 * delta_x2, 0)
    spring1_lock = IfElse((delta_x1_raw > scaled_thresh1) & (delta_x2_raw < scaled_thresh2), scale2 * delta_x, 0)
    spring2_lock = IfElse((delta_x1_raw < scaled_thresh1) & (delta_x2_raw > scaled_thresh2), scale1 * delta_x, 0)
    both_lock = IfElse((delta_x1_raw > scaled_thresh1) & (delta_x2_raw > scaled_thresh2), g, 0)

    e = IfElse(delta_x < 0, 0, neither_lock + spring1_lock + spring2_lock + both_lock)

    return e


def solve_for_time_given_position(nadir: Const, apex: Const, threshold1: Var, threshold2: Var, scale1: Var, scale2: Var):
    """The time at which the bungee system goes from the apex to the nadir,
    assuming that is initially at rest position (0,0) and velocity. """
    disp = TegVar('disp')
    x_hat = TegVar('x_hat')
    g = 10

    # Solution to the second-order linear ODE
    inner_integrand = stress(disp, threshold1, threshold2, scale1, scale2) - g
    velocity = -2 * Teg(0, x_hat, inner_integrand, disp)
    expr = InvertSqrt(velocity)
    ode_solution_wrt_time = Teg(nadir, apex, expr, x_hat)

    return ode_solution_wrt_time, expr


def optimize(nadir: Const, apex: Const, threshold1, threshold2, scale1: Var, scale2: Var):
    """Optimizing both yield strength and scale for springiness. """
    g = 10
    # mass = 1
    num_samples = 10
    loss_values = []
    scale1_values = []
    scale2_values = []
    threshold1_values = []
    threshold2_values = []

    expr, invert_sqrt_vel = solve_for_time_given_position(nadir, apex, threshold1, threshold2, scale1, scale2)
    expr = simplify(expr)
    deriv = simplify(RevDeriv(expr, Tup(Const(1))))

    def loss_and_grads(args):
        scale1_val, scale2_val, threshold1_val, threshold2_val = args
        scale1.value = scale1_val
        scale2.value = scale2_val
        threshold1.value = threshold1_val
        threshold2.value = threshold2_val

        loss = evaluate(expr, num_samples=num_samples, ignore_cache=True)
        vel_blows = invert_sqrt_vel.blow_up
        big_loss_const = 50
        loss = big_loss_const if vel_blows else loss
        invert_sqrt_vel.blow_up = False

        grads = evaluate(deriv, num_samples=num_samples, ignore_cache=True)
        grads = (1, 1) if vel_blows else grads
        print(f'loss: {loss}')
        # print(f'scale: {scale_val}\n')
        loss_values.append(loss)
        scale1_values.append(scale1.value)
        scale2_values.append(scale2.value)
        threshold1_values.append(threshold1.value)
        threshold2_values.append(threshold2.value)
        return loss, grads

    def max_acceleration_bounded(args):
        # max(|acc|) < g
        # since acc < 0, -acc >= 0
        # max(-acc) < g
        # max(-(stress(disp) - g)) < g
        # max(-stress(disp)) < 0
        # -max(-stress(disp)) > 0
        # min(stress) > 0
        scale1, scale2, threshold1, threshold2 = args
        min_stress = min([evaluate(stress(Const(x), threshold1, threshold2, scale1, scale2), num_samples=num_samples, ignore_cache=True)
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
        scale1, scale2, threshold1, threshold2 = args

        def vel(x_hat):
            disp = TegVar('disp')
            inner_integrand = stress(disp, threshold1, threshold2, scale1, scale2) - g
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

    options = {'maxiter': 5}
    print('Starting minimization')
    minimize(loss_and_grads, [scale1.value, scale2.value, threshold1.value, threshold2.value], constraints=cons, tol=1e-1, jac=True, options=options)
    print('Ending minimization')
    return loss_values, scale1_values, scale2_values, threshold1_values, threshold2_values


if __name__ == "__main__":
    # Parameters to optimize
    scale1_init = 2
    scale2_init = 1
    threshold1_init = 1
    threshold2_init = 1
    scale1 = Var('scale1', scale1_init)
    scale2 = Var('scale2', scale2_init)
    threshold1 = Var('threshold1', threshold1_init)
    threshold2 = Var('threshold2', threshold2_init)
    nadir = Const(1e-5)
    apex = Const(5)

    loss_values, scale1_values, scale2_values, threshold1_values, threshold2_values = optimize(nadir, apex, threshold1, threshold2, scale1, scale2)

    import numpy as np
    strains = np.arange(0, apex.value, step=0.01)
    stresses_before = [evaluate(stress(Const(strain), threshold1_init, threshold1_init, scale1_init, scale2_init), ignore_cache=True)
                       for strain in strains]
    stresses_after = [evaluate(stress(Const(strain), threshold1_values[-1], threshold2_values[-1], scale1_values[-1], scale2_values[-1]), ignore_cache=True)
                      for strain in strains]
    print(scale1_values, scale2_values, threshold1_values, threshold2_values)

    fig, axes = plt.subplots(nrows=2, ncols=3)
    plat = plt
    plt = axes[0][0]

    plt.plot(strains, stresses_before, label='Before')
    plt.plot(strains, stresses_after, label='After')
    plt.set(xlabel='Strain', ylabel='Stress')
    # plt.legend()

    plt = axes[0][1]
    plt.plot(loss_values)
    plt.set(xlabel='Iteration of Optimization', ylabel='Value of Loss')

    plt = axes[1][0]
    plt.plot(scale1_values)
    plt.set(xlabel='Iteration of Optimization', ylabel='Scale1 of Stress')

    plt = axes[1][1]
    plt.plot(scale2_values)
    plt.set(xlabel='Iteration of Optimization', ylabel='Scale2 of Stress')

    plt = axes[0][2]
    plt.plot(threshold1_values)
    plt.set(xlabel='Iteration of Optimization', ylabel='Threshold1')

    plt = axes[1][2]
    plt.plot(threshold2_values)
    plt.set(xlabel='Iteration of Optimization', ylabel='Threshold2')

    plat.show()

b = 3
s = 5
h = 5
