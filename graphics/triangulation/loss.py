from teg import (
    Const,
    Var,
    TegVar,
    IfElse,
    Teg,
    Tup
)

from teg.math import (
    Sqr
)


from teg.derivs.reverse_deriv import reverse_deriv
from teg.passes.reduce import reduce_to_base
from teg.passes.simplify import simplify


def make_teg_program():
    # Triangle vertices
    tx1 = Var("tx1", 2)
    tx2 = Var("tx2", 2)
    tx3 = Var("tx3", 2)

    ty1 = Var("ty1", 2)
    ty2 = Var("ty2", 2)
    ty3 = Var("ty3", 2)

    # Pixel boundaries
    px0 = Var("px0", 2)
    px1 = Var("px1", 2)

    py0 = Var("py0", 2)
    py1 = Var("py1", 2)

    # Pixel color
    c0, c1, c2 = (Var("pc0", 0), Var("pc1", 0), Var("pc2", 0))

    # Triangle color
    tc0, tc1, tc2 = (Var("tc0", 0), Var("tc1", 0), Var("tc2", 0))

    # Variables of integration.
    x = TegVar("x")
    y = TegVar("y")

    line12_determinant = (tx1 * ty2 - tx2 * ty1) + (ty1 - ty2) * x + (tx2 - tx1) * y
    line23_determinant = (tx2 * ty3 - tx3 * ty2) + (ty2 - ty3) * x + (tx3 - tx2) * y
    line31_determinant = (tx3 * ty1 - tx1 * ty3) + (ty3 - ty1) * x + (tx1 - tx3) * y

    # Build active mask.
    point_mask = (IfElse(line12_determinant < 0, 1, 0) *
                  IfElse(line23_determinant < 0, 1, 0) *
                  IfElse(line31_determinant < 0, 1, 0))

    point_loss = Sqr(tc0 - c0) + Sqr(tc1 - c1) + Sqr(tc2 - c2)

    integral_x = Teg(px0, px1, point_loss * point_mask, x)
    integral_xy = Teg(py0, py1, integral_x, y)

    # Specify argument ordering.
    arglist = (tx1, ty1, tx2, ty2, tx3, ty3,
               px0, px1,
               py0, py1,
               c0, c1, c2,
               tc0, tc1, tc2)

    return integral_xy, arglist


__PROGRAM__, __ARGLIST__ = make_teg_program()
