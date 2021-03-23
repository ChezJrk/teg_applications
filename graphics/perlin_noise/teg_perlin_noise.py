from teg_perlin_base import make_perlin_integrals

def make_teg_program():
    _, (integral_noise, arglist), _, _, _ = make_perlin_integrals()
    return integral_noise, arglist

__PROGRAM__, __ARGLIST__ = make_teg_program()