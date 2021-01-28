from teg_perlin_base import make_perlin_integrals

def make_teg_program():
    (integral_thresholded, arglist), _, _, _, _ = make_perlin_integrals()
    return integral_thresholded, arglist

__PROGRAM__, __ARGLIST__ = make_teg_program()