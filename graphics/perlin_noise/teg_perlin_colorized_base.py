from teg import (
    Const,
    Var,
    TegVar,
    IfElse,
    Teg,
    Tup,
    LetIn
)

from teg.math import (
    Sqr
)

from teg.maps.smoothstep import teg_smoothstep
from functools import partial

def letify(*exprs):
    new_vars = [Var(f'_{idx}') for idx, expr in enumerate(exprs)]
    return new_vars, partial(LetIn, new_vars, exprs)

def make_perlin_integrals():
    # Quad boundaries
    grid_x0 = Var("gx0", 0)
    grid_x1 = Var("gx1", 1)
    grid_y0 = Var("gy0", 0)
    grid_y1 = Var("gy1", 1)

    # Boundary vectors.
    vec_x00 = Var('vx00x', 0), Var('vx00y', 0)
    vec_x10 = Var('vx10x', 0), Var('vx10y', 0)
    vec_x01 = Var('vx01x', 0), Var('vx01y', 0)
    vec_x11 = Var('vx11x', 0), Var('vx11y', 0)

    col_00 = Var('col00r', 0), Var('col00g', 0), Var('col00b', 0)
    col_10 = Var('col10r', 0), Var('col10g', 0), Var('col10b', 0)
    col_01 = Var('col01r', 0), Var('col01g', 0), Var('col01b', 0)
    col_11 = Var('col11r', 0), Var('col11g', 0), Var('col11b', 0)

    cols = (col_00, col_10, col_01, col_11)
    vecs = (vec_x00, vec_x10, vec_x01, vec_x11)

    # Pixel boundaries
    px0 = Var("px0", 2)
    px1 = Var("px1", 2)

    py0 = Var("py0", 2)
    py1 = Var("py1", 2)

    # Pixel colors.
    pix = Var('pix_r', 0), Var('pix_g', 0), Var('pix_b', 0)

    # Negative-region color
    nc = (Var("nc0", 0), Var("nc1", 0), Var("nc2", 0))

    # Region threshold value(s).
    t = Var('t')
    t2 = Var('t2')

    # Variables of integration.
    x = TegVar("x")
    y = TegVar("y")

    def interpolate(val00, val10, val01, val11, interp_x, interp_y):
        return ((val00) * (1 - interp_x) + (val10) * (interp_x)) * (1 - interp_y) +\
               ((val01) * (1 - interp_x) + (val11) * (interp_x)) * (interp_y)

    def dot(xs, ys):
        return sum(_x * _y for _x, _y in zip(xs, ys))

    """
    def perlin_contribs_at(x0, y0, x1, y1, interp_x, interp_y, vecs):
        # Compute interpolated perlin value at the four corners of the pixel.
        return tuple(interpolate(*[dot(c_dist, vec) for vec in vecs],
                                 interp_x, interp_y) for c_dist in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)])
    """

    def perlin_contrib_at(_x, _y, grid_bounds, grid_vecs):
        x0 = (_x - grid_bounds[0][0]) / (grid_bounds[1][0] - grid_bounds[0][0])
        y0 = (_y - grid_bounds[0][1]) / (grid_bounds[1][1] - grid_bounds[0][1])
        x1 = (_x - grid_bounds[1][0]) / (grid_bounds[1][0] - grid_bounds[0][0])
        y1 = (_y - grid_bounds[1][1]) / (grid_bounds[1][1] - grid_bounds[0][1])

        s_x0, s_y0 = teg_smoothstep(x0), teg_smoothstep(y0)
        return interpolate(dot((x0, y0), grid_vecs[0][0]),
                           dot((x1, y0), grid_vecs[1][0]),
                           dot((x0, y1), grid_vecs[0][1]),
                           dot((x1, y1), grid_vecs[1][1]),
                           s_x0,
                           s_y0)

    def colorization_at(_x, _y, grid_bounds, grid_cols):
        x0 = (_x - grid_bounds[0][0]) / (grid_bounds[1][0] - grid_bounds[0][0])
        y0 = (_y - grid_bounds[0][1]) / (grid_bounds[1][1] - grid_bounds[0][1])
        x1 = (_x - grid_bounds[1][0]) / (grid_bounds[1][0] - grid_bounds[0][0])
        y1 = (_y - grid_bounds[1][1]) / (grid_bounds[1][1] - grid_bounds[0][1])

        #s_x0, s_y0 = teg_smoothstep(x0), teg_smoothstep(y0)
        return interpolate(grid_cols[0],
                           grid_cols[1],
                           grid_cols[2],
                           grid_cols[3],
                           x0,
                           y0)

    grid_bounds = ((grid_x0, grid_y0), (grid_x1, grid_y1))
    grid_vecs = ((vec_x00, vec_x01), (vec_x10, vec_x11))
    c00 = perlin_contrib_at(px0, py0, grid_bounds, grid_vecs)
    c10 = perlin_contrib_at(px1, py0, grid_bounds, grid_vecs)
    c01 = perlin_contrib_at(px0, py1, grid_bounds, grid_vecs)
    c11 = perlin_contrib_at(px1, py1, grid_bounds, grid_vecs)

    # (c00, c10, c01, c11) = perlin_contribs_at(x0, y0, x1, y1, s_x0, s_y1, vecs)

    # Use let expressions for quicker compilation and smaller programs.
    (c00, c10, c01, c11), c_let_expr = letify(c00, c10, c01, c11)

    bilinear_lerp = (c00 * (px1 - x) + c10 * (x - px0)) * (py1 - y) +\
                    (c01 * (px1 - x) + c11 * (x - px0)) * (y - py0)

    bilinear_lerp = bilinear_lerp / ((px1 - px0) * (py1 - py0))

    pc0 = colorization_at(x, y,
                          grid_bounds,
                          (cols[0][0],
                          cols[1][0],
                          cols[2][0],
                          cols[3][0]))
    pc1 = colorization_at(x, y,
                          grid_bounds,
                          (cols[0][1],
                          cols[1][1],
                          cols[2][1],
                          cols[3][1]))
    pc2 = colorization_at(x, y,
                          grid_bounds,
                          (cols[0][2],
                          cols[1][2],
                          cols[2][2],
                          cols[3][2]))

    pc = (pc0, pc1, pc2)
    # Derivative of threshold only.
    thresholded_integral = Teg(px0, px1,
                               Teg(py0, py1,
                                   c_let_expr(
                                        IfElse(bilinear_lerp > t, Tup(*pc), Tup(*nc))
                                        #IfElse(bilinear_lerp > t, pc[2] * pc[1] * pc[0], nc[2] * nc[1] * nc[0])
                                    ), y
                                   ), x
                               )

    point_loss_pc = Sqr(pc[0] - pix[0]) + Sqr(pc[1] - pix[1]) + Sqr(pc[2] - pix[2])
    point_loss_nc = Sqr(nc[0] - pix[0]) + Sqr(nc[1] - pix[1]) + Sqr(nc[2] - pix[2])

    loss_integral = Teg(px0, px1,
                        Teg(py0, py1,
                            c_let_expr(
                                IfElse(bilinear_lerp > t, point_loss_pc, point_loss_nc)
                                ), y
                            ), x
                        )

    noise_integral = Teg(px0, px1,
                         Teg(py0, py1,
                             c_let_expr(
                                    bilinear_lerp
                                    ), y
                             ), x
                        )

    l_arglist = (*vec_x00, *vec_x10, *vec_x01, *vec_x11,
                 grid_x0, grid_x1,
                 grid_y0, grid_y1,
                 px0, px1,
                 py0, py1,
                 *col_00, *col_10, *col_01, *col_11,
                 *nc, *pix,
                 t)

    t_arglist = (*vec_x00, *vec_x10, *vec_x01, *vec_x11,
                 grid_x0, grid_x1,
                 grid_y0, grid_y1,
                 px0, px1,
                 py0, py1,
                 *col_00, *col_10, *col_01, *col_11,
                 *nc, t)

    n_arglist = (*vec_x00, *vec_x10, *vec_x01, *vec_x11,
                 grid_x0, grid_x1,
                 grid_y0, grid_y1,
                 px0, px1,
                 py0, py1)

    return ((thresholded_integral, t_arglist),
            (noise_integral, n_arglist),
            (loss_integral, l_arglist))
