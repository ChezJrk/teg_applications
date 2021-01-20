from teg_linear_base import make_linear_integral


def make_teg_program():
    _, (integral_xy, arglist) = make_linear_integral()

    return integral_xy, arglist


__PROGRAM__, __ARGLIST__ = make_teg_program()
