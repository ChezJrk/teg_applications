import matplotlib.pyplot as plt
import numpy as np

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
from utils import normal_pdf


def mse(y_pred, y_target):
    return (y_pred - y_target)**2


def generate_artificial_data():
    x = TegVar('x')
    mu1 = Const(name='mu1', value=25)
    mu2 = Const(name='mu2', value=25)
    sigma1 = Const(name='sigma1', value=8.1)
    sigma2 = Const(name='sigma2', value=1.5)
    i1 = Const(name='i1', value=2.02)
    i2 = Const(name='i2', value=1)
    a1 = Var('a1', 25)
    a2 = Var('a2', 1)

    pdf_ends = -a1 * normal_pdf(x, mu1, sigma1) + i1
    pdf_center = a2 * normal_pdf(x, mu2, sigma2) + i2

    lower_threshold = Var('lower_threshold', 20)
    upper_threshold = Var('upper_threshold', 30)

    predictor = IfElse((lower_threshold < x) & (x < upper_threshold), pdf_center, pdf_ends)

    # Generate samples for the model
    npixels_intensity = {}
    for i in range(50):
        x.value = i
        npixels_intensity[x.value] = evaluate(predictor, ignore_cache=True)

    return npixels_intensity


def spline_data_from_points(data):
    def spline_func(x):
        expr = Const(0)
        for i in range(49):
            left = data[i]
            right = data[i+1]
            linear_interp = left * (1 - (x - i)) + right * (x - i)
            expr += IfElse((i <= x) & (x < i + 1), linear_interp, Const(0))
        return expr

    return spline_func


def vein_artery_classifier(continuous_data):
    x = TegVar('x')
    mu1 = Const(name='mu1', value=25)
    mu2 = Const(name='mu2', value=25)
    sigma1 = Const(name='sigma1', value=8.1)
    sigma2 = Const(name='sigma2', value=1.5)
    i1 = Const(name='i1', value=2.02)
    i2 = Const(name='i2', value=1)
    a1 = Var('a1', 25)
    a2 = Var('a2', 1)

    pdf_ends = -a1 * normal_pdf(x, mu1, sigma1) + i1
    pdf_center = a2 * normal_pdf(x, mu2, sigma2) + i2

    lower_threshold = Var('lower_threshold', 21)
    upper_threshold = Var('upper_threshold', 30)

    predictor = IfElse((lower_threshold < x) & (x < upper_threshold), pdf_center, pdf_ends)

    loss = Teg(0, 50, mse(continuous_data(x), predictor), x)

    # Optimize by gradient descent
    num_samples = 100
    loss_values = [evaluate(loss, num_samples=num_samples, ignore_cache=True)]
    lower_threshold_values = [lower_threshold.value]
    upper_threshold_values = [upper_threshold.value]
    for i in range(20):
        da1, da2, dlower, dupper = evaluate(RevDeriv(loss, Tup(Const(1))), num_samples=num_samples, ignore_cache=True)
        step_size = 0.25
        # a1 -= step_size * da1
        # a2 -= step_size * da2
        lower_threshold.value -= step_size * dlower
        upper_threshold.value -= step_size * dupper
        loss_values.append(evaluate(loss, num_samples=num_samples,  ignore_cache=True))
        lower_threshold_values.append(lower_threshold.value)
        upper_threshold_values.append(upper_threshold.value)

    return loss_values, lower_threshold_values, upper_threshold_values


if __name__ == "__main__":
    npixels_intensity = generate_artificial_data()
    # print(npixels_intensity)
    # npixels_intensity = {i: i**2 for i in range(-10, 12)}
    continuous_data = spline_data_from_points(npixels_intensity)
    loss_values, lower_threshold_values, upper_threshold_values = vein_artery_classifier(continuous_data)
    print(loss_values)
    print(lower_threshold_values)
    print(upper_threshold_values)
    # plt.plot(loss_values)
    # plt.ylabel('Value of Loss')
    # plt.xlabel('Iteration of Gradient Descent')
    # plt.show()

    # plt.plot(lower_threshold_values)
    # plt.ylabel('Lower Threshold')
    # plt.xlabel('Iteration of Gradient Descent')
    # plt.show()

    # plt.plot(lower_threshold_values)
    # plt.ylabel('Upper Threshold')
    # plt.xlabel('Iteration of Gradient Descent')
    # plt.show()
