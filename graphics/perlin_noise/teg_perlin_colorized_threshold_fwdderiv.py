from teg_perlin_colorized_base import make_perlin_integrals
#from teg.derivs.reverse_deriv import reverse_deriv
from teg.derivs.fwd_deriv import fwd_deriv
from teg.passes.simplify import simplify
from teg.passes.reduce import reduce_to_base

from teg import (Tup, Const)
def make_teg_program():
    _, _, (loss_integral, arglist) = make_perlin_integrals()
    (vec_00x, vec_00y, vec_10x, vec_10y,
     vec_01x, vec_01y, vec_11x, vec_11y,
     grid_x0, grid_x1,
     grid_y0, grid_y1,
     px0, px1,
     py0, py1,
     c00r, c00g, c00b,
     c10r, c10g, c10b,
     c01r, c01g, c01b,
     c11r, c11g, c11b,
     nc0, nc1, nc2,
     pix_r, pix_g, pix_b, t) = arglist

    output_list = (vec_00x, vec_00y, vec_10x, vec_10y, 
                   vec_01x, vec_01y, vec_11x, vec_11y,
                   c00r, c00g, c00b,
                   c10r, c10g, c10b,
                   c01r, c01g, c01b,
                   c11r, c11g, c11b,
                   nc0, nc1, nc2, t)

    derivs = []
    for output in output_list:
        print(f'Constructing derivative for: {output.name}')
        _deriv = fwd_deriv(loss_integral, [(output, 1)], replace_derivs=True)
        deriv = reduce_to_base(simplify(_deriv))
        derivs.append(deriv)

    return Tup(*derivs), arglist

__PROGRAM__, __ARGLIST__ = make_teg_program()