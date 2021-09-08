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

    (tx1, ty1, tx2, ty2, tx3, ty3,
     px0, px1,
     py0, py1,
     c0, c1, c2,
     tc00, tc10, tc20,
     tc01, tc11, tc21,
     tc02, tc12, tc22) = arglist

    output_list = [tx1, tx2, tx3,
                   ty1, ty2, ty3,
                   tc00, tc10, tc20,
                   tc01, tc11, tc21,
                   tc02, tc12, tc22]

    _,  integral = reverse_deriv(integral_xy, Tup(Const(1)), output_list=output_list, args={'ignore_deltas': True})
    integral = reduce_to_base(simplify(integral))

    # Specify argument ordering.
    arglist = (tx1, ty1, tx2, ty2, tx3, ty3,
               px0, px1,
               py0, py1,
               c0, c1, c2,
               tc00, tc10, tc20,
               tc01, tc11, tc21,
               tc02, tc12, tc22)

    return integral, arglist


__PROGRAM__, __ARGLIST__ = make_teg_program()
