from typing import List, Tuple
import time
import collections
import itertools
import math
import numpy as np
import scipy as sp
import scipy.optimize as spop
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
    And,
    Or,
    Bool,
)
from teg.math.smooth import (
    Sqr
)
from teg.lang.integrable_program import false as TegFalse
from teg.derivs import FwdDeriv, RevDeriv
from teg.eval import numpy_eval as evaluate_numpy
from teg.passes import simplify
from teg.passes import substitute
from teg.ir import emit
from tests.c_utils import runProgram, compileProgram
from tap import Tap


def remap(t, t0, t1, x0, x1):
    s = (t - t0) * (1 / (t1 - t0))
    return x0 + s*(x1 - x0)


# linear spline expression through xs; linear runtime
def linspline(t: ITeg, ts: List[ITeg], xs: List[ITeg]):
    # assert len(ts) == len(xs)
    # assert len(ts) > 0
    #
    # f = xs[0]
    # for pt, nt, px, nx in zip(ts, ts[1:], xs, xs[1:]):
    #     m = (px - nx) / (pt - nt)
    #     b = px - m * pt
    #     ramp = m * t + b
    #     f = IfElse(t < pt, f, ramp)
    # f = IfElse(t < ts[-1], f, xs[-1])
    # return f
    assert len(ts) == len(xs)
    assert len(ts) > 0

    f = Const(0)
    for idx, (t0, t1, x0, x1) in enumerate(zip(ts, ts[1:], xs, xs[1:])):
        indicator = t < t1 if idx == 0 else t0 <= t if idx == (len(ts) - 2) else And(t0 <= t, t < t1)
        lerp01 = remap(t, t0, t1, x0, x1)
        f += IfElse(indicator, lerp01, Const(0))
    return f


# linear spline expression through xs; linear runtime
def condition_linspline(t: ITeg, ts: List[ITeg], xs: List[ITeg], bool_func):
    assert len(ts) == len(xs)
    assert len(ts) > 0

    f = TegFalse
    for idx, (t0, t1, x0, x1) in enumerate(zip(ts, ts[1:], xs, xs[1:])):
        indicator = t < t1 if idx == 0 else t0 <= t if idx == (len(ts) - 2) else And(t0 <= t, t < t1)
        lerp01 = remap(t, t0, t1, x0, x1)
        f = Or(f, And(indicator, bool_func(lerp01)))

    return f


# cubic hermite spline expression through xs; linear runtime
def cubic_hermite_spline(t: ITeg, ts: List[ITeg], xs: List[ITeg], dxs:List[ITeg]):
    assert len(ts) == len(xs)
    assert len(ts) == len(dxs)
    assert len(ts) > 0

    f = xs[0]
    for idx, (pt, nt, px, nx, pdx, ndx), in enumerate(zip(ts, ts[1:], xs, xs[1:], dxs, dxs[1:])):
        u = (t - pt) / (nt - pt)  # == t normalized to [0, 1]
        u2 = u * u
        u3 = u * u2
        ramp = (2 * u3 - 3 * u2 + 1) * px + (u3 - 2 * u2 + u) * pdx + (-2 * u3 + 3 * u2) * nx + (u3 - u2) * ndx
        f = ramp if idx == 0 else IfElse(t < pt, f, ramp)
    return f


def catmull_spline(t: ITeg, ts: List[ITeg], xs: List[ITeg]):
    assert len(ts) == len(xs)
    assert len(ts) > 0

    dxs = []
    ts_ = [ts[0], *ts, ts[-1]]
    xs_ = [xs[0], *xs, xs[-1]]
    for idx in range(1, len(ts) + 1):
        pt = ts_[idx - 1]
        nt = ts_[idx + 1]
        px = xs_[idx - 1]
        nx = xs_[idx + 1]
        dxs.append((nx - px) / (nt - pt))

    return cubic_hermite_spline(t, ts, xs, dxs)


def test_func(t, a, m):
    args = Args()
    t_ = TegVar('t_')

    # a_ = a
    # a = t
    params = [a]

    ts = [0, 1, 2]
    xs = [0, a, 0]
    x = linspline(t_, ts, xs)
    x_cond = condition_linspline(t_, ts, xs, lambda expr: (1 < expr))
    k = 10000
    potential = k * IfElse(x_cond, Sqr(x - 1), 0)
    # potential = k * IfElse(x_cond, 1, 0)

    v = FwdDeriv(x, [(t_, 1), *[(_, 0) for _ in params]]).deriv_expr
    kinetic = 1/2*m*v*v
    lagrangian = kinetic - potential
    action = Teg(0, 2, lagrangian, t_)
    dSda = RevDeriv(action, Tup(Const(1)))
    dels = dSda.variables
    print(dels)

    def bind_param(expr, vs):
        for param, v in zip(params, vs):
            expr.bind_variable(param, v)

    def action_func(vs):
        bind_param(action, vs)
        return evaluate_numpy(-action, num_samples=args.num_samples, ignore_cache=True)

    def d_action_func(vs):
        bind_param(dSda, vs)
        k = evaluate_numpy(dSda, num_samples=args.num_samples, ignore_cache=True)
        return k[0]

    init_guess = [2.0]
    # init_guess = [1.03571428]
    print(d_action_func(init_guess))
    print(action_func(init_guess))
    # res = spop.root_scalar(d_action_func, bracket=[0.1, 2.0], method='bisect')
    res = spop.root(d_action_func, x0=init_guess, method='hybr', tol=1e-8)
    # res = spop.minimize(action_func, init_guess, method='Nelder-Mead', tol=1e-12)
    # res = spop.minimize(action_func, init_guess, method='CG', tol=1e-6, jac=loss_grad)

    print(res.x)
    print(d_action_func(res.x))
    x.bind_variable(a, res.x)
    return substitute.substitute(x, t_, t)
    # return action
    # return dSda


class Args(Tap):
    pixel_width: int = 30
    pixel_height: int = 30
    num_samples: int = 30
    t_samples: int = 2000


def main():
    args = Args()
    fig, axes = plt.subplots(nrows=1, ncols=2)

    a0 = 1.5
    m0 = 1
    t = Var('t')
    a = Var('a', a0)
    m = Var('m', m0)
    param_vars = [
        a,
        m,
    ]
    differential_vals = collections.defaultdict(int, {
        'a': 1,
        'm': 0,
    })

    teg_func = test_func

    def sample_expr(expr, ts):
        for t_ in ts:
            print(t_)
            expr.bind_variable(t, t_)
            f = evaluate_numpy(expr, num_samples=args.num_samples, ignore_cache=True)
            # print(t_, f)
            yield f

    t_samples = 40
    def map_sample_func(f, ts):
        return Tup(*[
            f(Const(t), a, m)
            for t in ts
        ])

    ts = [(2*i)/args.t_samples for i in range(args.t_samples + 1)]
    fexpr = teg_func(t, a, m)
    xs = list(sample_expr(fexpr, ts))

    axes[0].plot(ts, xs)
    # axes[1].plot(ts, dxs)




    plt.show()



if __name__ == '__main__':
    main()
