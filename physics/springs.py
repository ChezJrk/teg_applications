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
    FwdDeriv,
)
from teg.math.smooth import Sqrt
from teg.passes.evaluate import evaluate


def stress(strain: ITeg) -> ITeg:
    yield_strength = 10
    stress = IfElse(strain < yield_strength, 2 * strain, strain**2 / 5)
    return stress


def solve_for_time_given_position(position: Const) -> ITeg:
    x = TegVar('x')
    x_hat = TegVar('x_hat')
    x_hat_hat = Const(name='x_hat_hat', value=position.value)
    c0 = Const(10)
    c1 = Const(10)
    expr = 1 / Sqrt(2 * Teg(0, x_hat, -stress(x), x) + c0)
    ode_solution_wrt_time = Teg(0, x_hat_hat, expr, x_hat) + c1
    return ode_solution_wrt_time


def optimize_scale():
    scale = Var('scale', 1)
    ground_height = 100
    position = Const(1)
    penalty_value = Const(10)
    penalty = IfElse(x < ground_height, 0, penalty_value)
    loss = solve_for_time_given_position(position) + penalty

    # Optimize by gradient descent
    num_samples = 50
    loss_values = [evaluate(loss, num_samples=num_samples, ignore_cache=True)]
    scale_values = [scale.value]
    for i in range(25):
        dscale = evaluate(FwdDeriv(loss, (scale, 1)), num_samples=num_samples, ignore_cache=True)
        step_size = 0.5
        scale.value -= step_size * dscale
        loss_values.append(evaluate(loss, num_samples=num_samples, ignore_cache=True))
        scale_values.append(scale.value)

    return loss_values, scale_values


if __name__ == "__main__":
    loss_values, scale_values = optimize_scale()

    plt.plot(loss_values)
    plt.ylabel('Value of Loss')
    plt.xlabel('Iteration of Gradient Descent')
    plt.show()

    plt.plot(scale_values)
    plt.ylabel('Scale of Stress')
    plt.xlabel('Iteration of Gradient Descent')
    plt.show()
