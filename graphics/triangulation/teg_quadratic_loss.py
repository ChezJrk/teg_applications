from teg import (
    Tup,
    Const
)

from teg.derivs.reverse_deriv import reverse_deriv
from teg.passes.reduce import reduce_to_base
from teg.passes.simplify import simplify

from teg_quadratic_base import make_quadratic_integral


def make_teg_program():
    (integral_xy, arglist), _ = make_quadratic_integral()
    return integral_xy, arglist


__PROGRAM__, __ARGLIST__ = make_teg_program()
