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


def make_linear_bilinear_integral():
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

    # Pixel corner color
    c00r, c00g, c00b = (Var("pc00r", 0), Var("pc00g", 0), Var("pc00b", 0))
    c01r, c01g, c01b = (Var("pc01r", 0), Var("pc01g", 0), Var("pc01b", 0))
    c10r, c10g, c10b = (Var("pc10r", 0), Var("pc10g", 0), Var("pc10b", 0))
    c11r, c11g, c11b = (Var("pc11r", 0), Var("pc11g", 0), Var("pc11b", 0))

    # Tri-vertex colors
    tc00, tc10, tc20 = (Var("tc00", 0), Var("tc10", 0), Var("tc20", 0))
    tc01, tc11, tc21 = (Var("tc01", 0), Var("tc11", 0), Var("tc21", 0))
    tc02, tc12, tc22 = (Var("tc02", 0), Var("tc12", 0), Var("tc22", 0))

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

    # Barycentric interpolation
    alpha = (((x - tx2) * (y - ty3) - (x - tx3) * (y - ty2)))
    beta = (((x - tx3) * (y - ty1) - (x - tx1) * (y - ty3)))
    gamma = (((x - tx1) * (y - ty2) - (x - tx2) * (y - ty1)))

    norm = alpha + beta + gamma
    tc0 = (alpha * tc00 + beta * tc01 + gamma * tc02) / norm
    tc1 = (alpha * tc10 + beta * tc11 + gamma * tc12) / norm
    tc2 = (alpha * tc20 + beta * tc21 + gamma * tc22) / norm

    # Bilinear interpolation
    c0r = (x - px0) * c10r + (px1 - x) * c00r
    c0g = (x - px0) * c10g + (px1 - x) * c00g
    c0b = (x - px0) * c10b + (px1 - x) * c00b

    c1r = (x - px0) * c11r + (px1 - x) * c01r
    c1g = (x - px0) * c11g + (px1 - x) * c01g
    c1b = (x - px0) * c11b + (px1 - x) * c01b

    cr = (y - py0) * c1r + (py1 - y) * c0r
    cg = (y - py0) * c1g + (py1 - y) * c0g
    cb = (y - py0) * c1b + (py1 - y) * c0b

    point_loss = Sqr(tc0 - cr) + Sqr(tc1 - cg) + Sqr(tc2 - cb)
    point_color = Tup(tc0, tc1, tc2)

    integral_x = Teg(px0, px1, point_loss * point_mask, x)
    integral_xy = Teg(py0, py1, integral_x, y)

    color_integral_x = Teg(px0, px1, point_color * point_mask, x)
    color_integral_xy = Teg(py0, py1, color_integral_x, y)

    arglist = (tx1, ty1, tx2, ty2, tx3, ty3,
               px0, px1,
               py0, py1,

               c00r, c00g, c00b,
               c01r, c01g, c01b,
               c10r, c10g, c10b,
               c11r, c11g, c11b,

               tc00, tc10, tc20,
               tc01, tc11, tc21,
               tc02, tc12, tc22)

    color_arglist = (tx1, ty1, tx2, ty2, tx3, ty3,
                     px0, px1,
                     py0, py1,
                     tc00, tc10, tc20,
                     tc01, tc11, tc21,
                     tc02, tc12, tc22)

    return (integral_xy, arglist), (color_integral_xy, color_arglist)
