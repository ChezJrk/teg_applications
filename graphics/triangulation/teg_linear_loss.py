from teg import (
    Tup,
    Const
)

from teg.derivs.reverse_deriv import reverse_deriv
from teg.passes.reduce import reduce_to_base
from teg.passes.simplify import simplify

from teg_linear_base import make_linear_integral


def make_teg_program():
    (integral_xy, arglist), _ = make_linear_integral()
    return integral_xy, arglist


__PROGRAM__, __ARGLIST__ = make_teg_program()
