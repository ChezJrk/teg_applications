
import math

from smooth import Exp
from teg import (
    Teg,
    TegVar,
)
from teg.eval.numpy_eval import evaluate


def normal_pdf(x, mu, sigma):
    """Probability distribution function of the normal distribution. """
    c = math.sqrt(2 * math.pi)
    return Exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * c)


def truncated_gaussian_pdf(x, mu, sigma, a, b):
    """Probability density function of a truncated normal distribution. """
    numerator = normal_pdf(x, mu, sigma)
    t = TegVar('t')
    teg_a_to_b = Teg(a, b, normal_pdf(t, mu, sigma), t)
    denominator = sigma * teg_a_to_b
    return numerator / denominator


def optimized_truncated_gaussian_pdf(x, mu, sigma, a, b, num_samples):
    """Probability density function of a truncated normal distribution. """
    numerator = normal_pdf(x, mu, sigma)
    t = TegVar('t')
    teg_a_to_b = Teg(a, b, normal_pdf(t, mu, sigma), t)
    denominator = sigma * teg_a_to_b
    return numerator / evaluate(denominator, num_samples=num_samples, ignore_cache=True)