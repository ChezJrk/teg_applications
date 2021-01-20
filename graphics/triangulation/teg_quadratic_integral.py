from teg_quadratic_base import make_quadratic_integral


def make_teg_program():
    _, (integral_xy, arglist) = make_quadratic_integral()

    return integral_xy, arglist


__PROGRAM__, __ARGLIST__ = make_teg_program()
