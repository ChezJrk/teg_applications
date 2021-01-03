import matplotlib.pyplot as plt

from teg.lang.integrable_program import (
    ITeg,
    Const,
    Var,
    IfElse,
    Teg,
    Tup,
    LetIn,
    TegVar,
    RevDeriv,
)
from teg.math.smooth import Sqrt
from teg.passes.evaluate import evaluate


def stress(strain: ITeg, yield_strength: Var) -> ITeg:
    stress = IfElse(strain < yield_strength, 2 * strain, strain**2 / 5)
    return stress


def solve_for_time_given_position(position: Const, yield_strength: Var) -> ITeg:
    x = TegVar('x')
    x_hat = TegVar('x_hat')
    x_hat_hat = Const(name='x_hat_hat', value=position.value)
    c0 = Const(10)
    c1 = Const(10)
    expr = 1 / Sqrt(2 * Teg(0, x_hat, -stress(x, yield_strength), x) + c0)
    ode_solution_wrt_time = Teg(0, x_hat_hat, expr, x_hat) + c1
    return ode_solution_wrt_time


def optimize():
    """Optimizing both yield strength and scale for springiness. """
    # Parameters to optimize
    scale = Var('scale', 1)
    yield_strength = Var('yield_strength', 10)
    ground_height = 100

    # Set up the loss
    position = Const(1)
    penalty_value = Const(10)
    penalty = IfElse(x >= ground_height, penalty_value, 0)
    loss = solve_for_time_given_position(position, yield_strength) + penalty

    # Optimize by gradient descent
    num_samples = 10
    loss_values = [evaluate(loss, num_samples=num_samples, ignore_cache=True)]
    scale_values = [scale.value]
    yield_strength_values = [yield_strength.value]
    for i in range(25):
        dscale, dyield_strength = evaluate(RevDeriv(loss, Tup(Const(1))), num_samples=num_samples, ignore_cache=True)

        # Take gradient step
        step_size = 0.5
        yield_strength.value -= step_size * dyield_strength
        scale.value -= step_size * dscale

        # Save data for plotting
        loss_values.append(evaluate(loss, num_samples=num_samples, ignore_cache=True))
        scale_values.append(scale.value)
        yield_strength_values.append(yield_strength.value)

    return loss_values, scale_values, yield_strength_values


if __name__ == "__main__":
    loss_values, scale_values, yield_strength_values = optimize()

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
