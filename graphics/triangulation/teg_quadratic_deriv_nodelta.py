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

    (tx1, ty1, tx2, ty2, tx3, ty3,
     px0, px1,
     py0, py1,
     c0, c1, c2,
     tc00, tc10, tc20,
     tc01, tc11, tc21,
     tc02, tc12, tc22,
     tch00, tch10, tch20,
     tch01, tch11, tch21,
     tch02, tch12, tch22) = arglist

    deriv_list = [tx1, tx2, tx3,
                  ty1, ty2, ty3,
                  tc00, tc10, tc20,
                  tc01, tc11, tc21,
                  tc02, tc12, tc22,
                  tch00, tch10, tch20,
                  tch01, tch11, tch21,
                  tch02, tch12, tch22]

    _,  integral = reverse_deriv(integral_xy, Tup(Const(1)), output_list=deriv_list, args={'ignore_deltas': True})
    integral = reduce_to_base(simplify(integral))

    # Specify argument ordering.
    arglist = (tx1, ty1, tx2, ty2, tx3, ty3,
               px0, px1,
               py0, py1,
               c0, c1, c2,
               tc00, tc10, tc20,
               tc01, tc11, tc21,
               tc02, tc12, tc22,
               tch00, tch10, tch20,
               tch01, tch11, tch21,
               tch02, tch12, tch22)

    return integral, arglist


__PROGRAM__, __ARGLIST__ = make_teg_program()
