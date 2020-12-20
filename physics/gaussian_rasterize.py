import sys
sys.path.append("../Teg")  # TODO: hacky; actually engineer a proper setup.py
from typing import List, Tuple
import time
import collections
import numpy as np
import matplotlib.pyplot as plt

from teg.lang.integrable_program import (
    ITeg,
    Const,
    Var,
    IfElse,
    Teg,
    Tup,
    LetIn,
    TegVar,
)
from teg.math.smooth import (
    Sqr
)
from util.smooth import (
    Exp
)
from teg.derivs import FwdDeriv
from teg.eval import numpy_eval as evaluate_numpy
from teg.passes.simplify import simplify
from teg.ir import emit
from tests.c_utils import runProgram, compileProgram
from tap import Tap


class Args(Tap):
    pixel_width: int = 10
    pixel_height: int = 10
    num_samples: int = 30


def evaluate_c(expr: ITeg, num_samples=5000, ignore_cache=False, silent=False):
    pcount_before = time.perf_counter()
    c_code = emit(expr, target='C', num_samples=num_samples)
    pcount_after = time.perf_counter()
    if not silent:
        print(f'Teg-to-C emit time: {pcount_after - pcount_before:.3f}s')

    pcount_before = time.perf_counter()
    binary = compileProgram(c_code)
    pcount_after = time.perf_counter()
    if not silent:
        print(f'C compile time:     {pcount_after - pcount_before:.3f}s')

    pcount_before = time.perf_counter()
    value = runProgram(binary)
    pcount_after = time.perf_counter()
    if not silent:
        print(f'C exec time:        {pcount_after - pcount_before:.3f}s')

    return value


def evaluate(*args, **kwargs):
    fast_eval = kwargs.pop('fast_eval', False)
    if fast_eval:
        if not kwargs.get('silent', True):
            print('Evaluating in fast-mode: C')
        return evaluate_c(*args, **kwargs)
    else:
        if not kwargs.get('silent', True):
            print('Evaluating using numpy/python')
        # kwargs['ignore_cache'] = True
        return evaluate_numpy(*args, **kwargs)


def rasterize_triangles():
    x, y = TegVar('x'), TegVar('y')
    theta = Var('theta', 1)
    A = Var('a', 1)
    mux = Var('mx', 0.5)
    muy = Var('mx', 0.5)
    sigmax = Var('sx', 0.15)
    sigmay = Var('sy', 0.15)

    param_vars = [
        A,
        mux,
        muy,
        sigmax,
        sigmay,
    ]

    def right_triangle(x0, y0):
        """ ◥ with upper right corner at (x0, y0) """
        return (y < y0) & (x < x0 + theta) & (x - x0 + y - y0 + 0.75 + theta > 0)

    inside_front_cond = right_triangle(0.7, 0.7)

    # body = IfElse(inside_front_cond, 1, 0)
    # body = IfElse((y < .7 + theta) & (x < .7 + theta) & (.65 + theta < x + y), 1, 0)

    def gaussian(x_, y_, a, mx, my, sx, sy):
        # return a * Exp(-0.5 * Sqr((y_ - my) / sy)) / (sy * 2.5066) \
        #          * Exp(-0.5 * Sqr((x_ - mx) / sx)) / (sx * 2.5066)
        return a * Exp(-0.5 * (Sqr((x_ - mx) / sx) + Sqr((y_ - my) / sy))) / (sx * sy * 2 * np.pi)


    body = gaussian(x, y, A, mux, muy, sigmax, sigmay)
    # body = Sqr(x - mux)
    # body = (x - mux) * (x - mux)

    w, h = args.pixel_width, args.pixel_height
    inv_area = w * h
    pixel_expr = Tup(*[
        Teg(i / w, (i + 1) / w, Teg(j / h, (j + 1) / h, body, x), y) * inv_area
        for i in range(w)
        for j in range(h)
    ])
    return pixel_expr, param_vars


if __name__ == '__main__':
    args = Args().parse_args()

    fig, axes = plt.subplots(nrows=1, ncols=2)
    pixel_expr, param_vars = rasterize_triangles()

    pcount_before = time.perf_counter()

    w, h = args.pixel_width, args.pixel_height
    res = evaluate(pixel_expr, num_samples=args.num_samples)
    pixel_grid = np.array(res).reshape((w, h))
    # axes[0].imshow(pixel_grid[::-1, :], vmin=-1/(w * h), vmax=1/(w * h))
    axes[0].imshow(pixel_grid, origin='lower', extent=(0, 1, 0, 1))

    import math
    differential_vals = collections.defaultdict(int, {
        'a': 0,
        'mx': 0,
        'my': 0,
        'sx': 1,
        'sy': 0,
    })
    deriv_expr = FwdDeriv(pixel_expr, [(theta, differential_vals[theta.name]) for theta in param_vars])

    res = evaluate(deriv_expr, num_samples=args.num_samples)
    pixel_grid = np.array(res).reshape((w, h))
    # axes[1].imshow(pixel_grid[::-1, :], vmin=-0.05, vmax=0.05)
    axes[1].imshow(pixel_grid, origin='lower', extent=(0, 1, 0, 1))
    pcount_after = time.perf_counter()
    print(f'total:\t{pcount_after - pcount_before}')
    plt.show()

