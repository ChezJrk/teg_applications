import matplotlib.pyplot as plt

from teg import (
    Const,
    Var,
    IfElse,
    Teg,
    Tup,
    TegVar,
)
from teg.eval.numpy_eval import evaluate
from teg.derivs import RevDeriv
from utils import optimized_truncated_gaussian_pdf


def find_threshold(num_samples=1000):
    """Detect the threshold to distinguish samples from two distributions.

    More concretely, it maximizes
    int_{x=-100}^100 
        truncated_gaussian(mu=-1, sigma=1)1_{x < threshold}
        + truncated_gaussian(mu=1, sigma=1)1_{x > threshold}
    wrt the threshold using gradient ascent. 
    """
    x = TegVar('x')
    sigma = Const(name='sigma', value=1)
    mu1 = Const(name='mu1', value=-1)
    mu2 = Const(name='mu2', value=1)

    # Piecewise mixture
    cutoff = Var('cutoff', 0.75)
    a, b = -100, 100
    lower_dist = optimized_truncated_gaussian_pdf(x, mu1, sigma, a, b, num_samples)
    upper_dist = optimized_truncated_gaussian_pdf(x, mu2, sigma, a, b, num_samples)
    pdf = IfElse(x < cutoff, lower_dist, upper_dist)

    integral = Teg(-100, 100, pdf, x)

    # Optimize by gradient ascent
    integral_values = [evaluate(integral, num_samples=num_samples, ignore_cache=True)]
    cutoff_values = [cutoff.value]
    for i in range(25):
        dcutoff = evaluate(RevDeriv(integral, Tup(Const(1))), num_samples=num_samples, ignore_cache=True)
        step_size = 0.5
        cutoff.value += step_size * dcutoff
        integral_values.append(evaluate(integral, num_samples=num_samples, ignore_cache=True))
        cutoff_values.append(cutoff.value)

    return integral_values, cutoff_values


if __name__ == "__main__":
    integral_values, cutoff_values = find_threshold()

    plt.plot(integral_values)
    plt.ylabel('Value of Integral')
    plt.xlabel('Iteration of Gradient Ascent')
    plt.show()

    plt.plot(cutoff_values)
    plt.ylabel('Cutoff')
    plt.xlabel('Iteration of Gradient Ascent')
    plt.show()
