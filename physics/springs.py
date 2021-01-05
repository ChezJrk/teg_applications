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


def stress(strain: ITeg, yield_strength: Var, scale: Var) -> ITeg:
    """Stress curve given the strain as in Young's Modulus. """
    a, h = yield_strength, 2.2 * yield_strength

    # A line with slope 2 from 0 to a
    before_yield = 2 * strain

    # A parabola from a to 2a and symmetric about 3a / 2 that is less steep than the line
    after_yield = 4 * ((2 * a - h) / (a**2)) * (strain - 3 * a / 2)**2 + h

    # Discontinuous stress as described by Young's Modulus
    stress = IfElse(strain < yield_strength, before_yield, after_yield)

    return scale * stress


def solve_for_time_given_position(position: Const, yield_strength: Var, scale: Var):
    """The time at which the bungee is at a given position (height) assuming it begins
    at rest position (0,0) and velocity. """
    x = TegVar('x')
    x_hat = TegVar('x_hat')
    ground_height = 100
    penalty = 100
    gravity = 10

    # Solution to the second-order linear ODE
    inner_integrand = gravity - stress(x, yield_strength, scale)
    velocity = 2 * Teg(0, x_hat, inner_integrand, x)
    x_hat.value = 1
    vs = [ground_height, 0, 1, 2, 3,4 ,5 ,6, 7,8 ,9 , 10]
    vres = []
    for v in vs:
        x_hat.value = v
        vres.append(evaluate(velocity, ignore_cache=True))
    print(f'vels: {vres}')
    x_hat.value = None
    # expr = Invert(Sqrt(velocity))
    expr = InvertSqrt(velocity)
    ode_solution_wrt_time = Teg(0, position, expr, x_hat)
    # print(f'der: {evaluate(RevDeriv(ode_solution_wrt_time, Tup(Const(1))))}')

    # TODO: there's a nan problem here. IfElse( > 0, Teg(0, ground_height, inner_integrand, x), 0)
    # activate_penalty = IsNotNan(Invert(Sqrt(Teg(0, ground_height, inner_integrand, x))))

    return ode_solution_wrt_time  # + penalty * activate_penalty, activate_penalty


def optimize(scale: Var, yield_strength: Var):
    """Optimizing both yield strength and scale for springiness. """
    # Set up the loss
    position = Const(10)
    vhi = 40000
    # loss = solve_for_time_given_position(position, yield_strength, scale)

    num_samples = 20
    loss_values = []
    scale_values = []
    yield_strength_values = []

    def ineq_constr(args):
        scale, yield_strength = args
        scale, yield_strength = Var('scale', scale), Var('yield_strength', yield_strength)
        x = TegVar('x')
        ground_height = 100
        gravity = 10
        inner_integrand = gravity - stress(x, yield_strength, scale)
        velocity = 2 * Teg(0, ground_height, inner_integrand, x)

        # return -yield_strength.value + 2
        v = evaluate(velocity, num_samples=num_samples, ignore_cache=True)

        return vhi - v

    def ineq_constr2(args):
        scale, yield_strength = args
        scale, yield_strength = Var('scale', scale), Var('yield_strength', yield_strength)
        x = TegVar('x')
        ground_height = 100
        gravity = 10
        inner_integrand = gravity - stress(x, yield_strength, scale)
        velocity = 2 * Teg(0, ground_height, inner_integrand, x)

        # return -yield_strength.value + 2
        v = evaluate(velocity, num_samples=num_samples, ignore_cache=True)

        return vhi + v

    def scale_constr(args):
        scale, yield_strength = args
        return scale - 0.1

    def yield_strength_constr(args):
        scale, yield_strength = args
        return yield_strength - 0.1

    def jac(args):
        scale, yield_strength = args
        scale, yield_strength = Var('scale', scale), Var('yield_strength', yield_strength)
        x = TegVar('x')
        ground_height = 100
        gravity = 10
        inner_integrand = gravity - stress(x, yield_strength, scale)
        expr = simplify(Teg(0, ground_height, inner_integrand, x))
        grads = evaluate(simplify(RevDeriv(expr, Tup(Const(1)))), num_samples=num_samples, ignore_cache=True)
        return grads

    def func(args):
        scale, yield_strength = args
        scale, yield_strength = Var('scale', scale), Var('yield_strength', yield_strength)
        expr = simplify(solve_for_time_given_position(position, yield_strength, scale))

        loss = evaluate(expr, num_samples=num_samples, ignore_cache=True)
        grads = evaluate(simplify(RevDeriv(expr, Tup(Const(1)))), num_samples=num_samples, ignore_cache=True)
        loss_values.append(loss)
        scale_values.append(scale.value)
        yield_strength_values.append(yield_strength.value)
        ret = ineq_constr(args)
        # print(f'ineq_constr: {ret}')
        return loss, grads

    cons = [
        {'type': 'ineq', 'fun': ineq_constr},
        {'type': 'ineq', 'fun': ineq_constr2},
        {'type': 'ineq', 'fun': scale_constr},
        {'type': 'ineq', 'fun': yield_strength_constr},
    ]
    minimize(func, [scale.value, yield_strength.value], constraints=cons, jac=True)

    # Optimize by gradient descent
    # num_samples = 30
    # loss_values = []
    # scale_values = []
    # yield_strength_values = []
    # loss = simplify(loss)
    # for i in trange(40):
    #     # Save data for plotting
    #     loss_values.append(evaluate(loss, num_samples=num_samples, ignore_cache=True))
    #     scale_values.append(scale.value)
    #     yield_strength_values.append(yield_strength.value)

    #     # Derivative computation
    #     deriv = RevDeriv(loss, Tup(Const(1))).deriv_expr
    #     deriv = substitute(deriv, activate_penalty.blew_up, Const(0) if activate_penalty.errored else Const(1))
    #     dscale, dyield_strength = evaluate(simplify(deriv), num_samples=num_samples, ignore_cache=True)

    #     # Take gradient step
    #     step_size = 0.1
    #     print(dyield_strength, dscale)
    #     yield_strength.value -= step_size * dyield_strength
    #     scale.value -= step_size * dscale

    # loss_values.append(evaluate(loss, num_samples=num_samples, ignore_cache=True))
    # scale_values.append(scale.value)
    # yield_strength_values.append(yield_strength.value)

    return loss_values, scale_values, yield_strength_values


if __name__ == "__main__":
    # Parameters to optimize
    scale_init = 2
    yield_strength_init = 10
    scale = Var('scale', scale_init)
    yield_strength = Var('yield_strength', yield_strength_init)

    loss_values, scale_values, yield_strength_values = optimize(scale, yield_strength)

    import numpy as np
    strains = np.arange(0, yield_strength_init*2, step=0.1)
    stresses_before = [evaluate(stress(Const(strain), yield_strength_init, scale_init), ignore_cache=True)
                       for strain in strains]
    stresses_after = [evaluate(stress(Const(strain), yield_strength_values[-1], scale_values[-1]), ignore_cache=True)
                      for strain in strains]

    print(scale_values, yield_strength_values)
    plt.plot(strains, stresses_before, label='Before')
    plt.plot(strains, stresses_after, label='After')
    plt.ylabel('Stress')
    plt.xlabel('Strain')
    plt.legend()
    plt.show()

    plt.plot(loss_values)
    plt.ylabel('Value of Loss')
    plt.xlabel('Iteration of Gradient Descent')
    plt.show()

    plt.plot(scale_values)
    plt.ylabel('Scale of Stress')
    plt.xlabel('Iteration of Gradient Descent')
    plt.show()

    plt.plot(yield_strength_values)
    plt.ylabel('Yield Strength Threshold')
    plt.xlabel('Iteration of Gradient Descent')
    plt.show()
