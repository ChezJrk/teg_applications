from teg import (
    ITeg,
    Const,
    Var,
    SmoothFunc,
)

import numpy as np


class InvertSqrt(SmoothFunc):
    """y = x^(-0.5) with suppressed errors """
    def __init__(self, expr: ITeg, name: str = "InvertSqrt"):
        super(InvertSqrt, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        return Const(-1/2) * InvertSqrt(self.expr)**3 * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        return out_deriv_expr * Const(-1/2) * InvertSqrt(self.expr)**3

    def operation(self, in_value):
        if (np.array(in_value) > 0).all():
            return 1 / np.sqrt(in_value)
        return np.zeros_like(in_value)

    def output_size(input_size):
        return input_size


class IsNotNan(SmoothFunc):
    """y = x^(-0.5) with suppressed errors """
    def __init__(self, expr: ITeg, name: str = "Sqrt"):
        super(IsNotNan, self).__init__(expr=expr, name=name)
        self.blew_up = Var('blow_up', 1)
        self.errored = False

    def fwd_deriv(self, in_deriv_expr: ITeg):
        return self.blew_up * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        return out_deriv_expr * self.blew_up

    def operation(self, in_value):
        if np.isnan(in_value):
            self.errored = True
            return 0
        return 1

    def output_size(input_size):
        return input_size
