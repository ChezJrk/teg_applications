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


def make_quadratic_integral():
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

    # Tri-vertex colors
    tc00, tc10, tc20 = (Var("tc00", 0), Var("tc10", 0), Var("tc20", 0))
    tc01, tc11, tc21 = (Var("tc01", 0), Var("tc11", 0), Var("tc21", 0))
    tc02, tc12, tc22 = (Var("tc02", 0), Var("tc12", 0), Var("tc22", 0))

    # Half-edge colors.
    tch00, tch10, tch20 = (Var("tch00", 0), Var("tch10", 0), Var("tch20", 0))
    tch01, tch11, tch21 = (Var("tch01", 0), Var("tch11", 0), Var("tch21", 0))
    tch02, tch12, tch22 = (Var("tch02", 0), Var("tch12", 0), Var("tch22", 0))

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

    # Quadratic FEM interpolation
    alpha = (((x - tx2) * (y - ty3) - (x - tx3) * (y - ty2)))
    beta = (((x - tx3) * (y - ty1) - (x - tx1) * (y - ty3)))
    gamma = (((x - tx1) * (y - ty2) - (x - tx2) * (y - ty1)))
    norm = alpha + beta + gamma

    alpha = alpha / norm
    beta = beta / norm
    gamma = gamma / norm

    N_200 = (alpha) * (2*alpha - 1)
    N_020 = (beta) * (2*beta - 1)
    N_002 = (gamma) * (2*gamma - 1)

    N_011 = 4 * beta * gamma
    N_101 = 4 * alpha * gamma
    N_110 = 4 * alpha * beta

    tc0 = (N_200 * tc00 + N_020 * tc01 + N_002 * tc02 + N_011 * tch00 + N_101 * tch01 + N_110 * tch02)
    tc1 = (N_200 * tc10 + N_020 * tc11 + N_002 * tc12 + N_011 * tch10 + N_101 * tch11 + N_110 * tch12)
    tc2 = (N_200 * tc20 + N_020 * tc21 + N_002 * tc22 + N_011 * tch20 + N_101 * tch21 + N_110 * tch22)

    point_loss = Sqr(tc0 - c0) + Sqr(tc1 - c1) + Sqr(tc2 - c2)
    point_color = Tup(tc0, tc1, tc2)

    integral_x = Teg(px0, px1, point_loss * point_mask, x)
    integral_xy = Teg(py0, py1, integral_x, y)

    color_integral_x = Teg(px0, px1, point_color * point_mask, x)
    color_integral_xy = Teg(py0, py1, color_integral_x, y)

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

    color_arglist = (tx1, ty1, tx2, ty2, tx3, ty3,
                     px0, px1,
                     py0, py1,
                     tc00, tc10, tc20,
                     tc01, tc11, tc21,
                     tc02, tc12, tc22,
                     tch00, tch10, tch20,
                     tch01, tch11, tch21,
                     tch02, tch12, tch22)

    return (integral_xy, arglist), (color_integral_xy, color_arglist)
