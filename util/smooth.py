"""
    Custom SmoothFunc class demonstration

    TODO: Add emit_c function mapppings
"""

from teg import (
    ITeg,
    Const,
    SmoothFunc,
    Invert,
)

import numpy as np


class Exp(SmoothFunc):
    """
        y = e^x
    """
    def __init__(self, expr: ITeg, name: str = "Exp"):
        super(Exp, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        return Exp(self.expr) * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        return out_deriv_expr * Exp(self.expr)

    def operation(self, in_value):
        return np.exp(in_value)


class PowN(SmoothFunc):
    """
        y = x^1000
    """
    def __init__(self, expr: ITeg, n=2, name: str = "PowN"):
        super(PowN, self).__init__(expr=expr, name=name)
        self.n = n

    def fwd_deriv(self, in_deriv_expr: ITeg):
        if self.n == 0:
            return 0
        else:
            return self.n * PowN(self.expr, self.n - 1) * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        if self.n == 0:
            return 0
        else:
            return out_deriv_expr * self.n * PowN(self.expr, self.n - 1)

    def operation(self, in_value):
        return in_value**self.n


class Sin(SmoothFunc):
    """
        y = sin(x)
    """
    def __init__(self, expr: ITeg, name: str = "Sin"):
        super(Sin, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        return Cos(self.expr) * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        return out_deriv_expr * Cos(self.expr)

    def operation(self, in_value):
        return np.sin(in_value)


class Cos(SmoothFunc):
    """
        y = cos(x)
    """
    def __init__(self, expr: ITeg, name: str = "Cos"):
        super(Cos, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        return -Sin(self.expr) * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        return out_deriv_expr * -Sin(self.expr)

    def operation(self, in_value):
        return np.cos(in_value)





