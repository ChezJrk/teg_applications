from typing import List, Tuple, Optional, Union
import time
import collections
import itertools
import math
import numpy as np
import scipy as sp
import scipy.optimize as spop
import matplotlib.pyplot as plt
import physics.billiards_constraints as bc

from teg.lang.base import (
    ITeg,
    Const,
    Var,
    IfElse,
    Tup,
    LetIn,
    And,
    Or,
    Bool,
)
from teg.lang.teg import (
    Teg,
    TegVar,
)
from teg.lang.base import false as TegFalse
import teg.math.smooth as smooth
import physics.smooth as psmooth
from teg.derivs import FwdDeriv, RevDeriv
from teg.derivs.reverse_deriv import reverse_deriv
from teg.derivs.fwd_deriv import fwd_deriv
from teg.eval import evaluate
from teg.passes import simplify
from teg.passes.substitute import substitute
from tap import Tap
from teg.passes.simplify import simplify
from teg.passes.reduce import reduce_to_base


def remap(t, t0, t1, x0, x1):
    s = (t - t0) / (t1 - t0)
    return x0 + s*(x1 - x0)


def linspline(t: ITeg, ts: List[Union[ITeg, int]], xs: List[Union[ITeg, int]]):
    assert len(ts) == len(xs)
    assert len(ts) > 0

    f = Const(0)
    for idx, (t0, t1, x0, x1) in enumerate(zip(ts, ts[1:], xs, xs[1:])):
        indicator = t < t1 if idx == 0 else t0 <= t if idx == (len(ts) - 2) else And(t0 <= t, t < t1)
        lerp01 = remap(t, t0, t1, x0, x1)
        f += IfElse(indicator, lerp01, Const(0))
    return f


def linspline_action(t: ITeg, ts: List[Union[ITeg, int]], xs: List[Union[ITeg, int]], ys: List[Union[ITeg, int]], m: Union[ITeg, int]):
    assert len(ts) == len(xs)
    assert len(ts) == len(ys)
    assert len(ts) > 0

    def linspline01(t_, t01, x01):
        t0, t1 = t01
        x0, x1 = x01
        return remap(t_, t0, t1, x0, x1)

    def lagrangian01(t_, idx):
        x = linspline01(t_, ts[idx:idx+2], xs[idx:idx+2])
        y = linspline01(t_, ts[idx:idx+2], ys[idx:idx+2])
        vx = simplify(reduce_to_base(fwd_deriv(x, [(t_, 1)])))
        vy = simplify(reduce_to_base(fwd_deriv(y, [(t_, 1)])))
        vx_ = Var('vx_')
        vy_ = Var('vy_')
        kinetic01 = 1/2 * m * LetIn([vx_, vy_], [vx, vy], vx_ * vx_ + vy_ * vy_)
        potential01 = 0
        return kinetic01 - potential01
    action = Const(0)
    for idx, (t0, t1) in enumerate(zip(ts, ts[1:])):
        action += Teg(t0, t1, lagrangian01(t, idx), t)
    return action


def cubspline_action(t: ITeg, ts: List[Union[ITeg, int]], xs: List[Union[ITeg, int]], ys: List[Union[ITeg, int]], m: Union[ITeg, int]):
    assert len(ts) == len(xs)
    assert len(ts) == len(ys)
    assert len(ts) > 0

    def cubspline01(t_, t01, x01, dx01):
        t0, t1 = t01
        x0, x1 = x01
        dx0, dx1 = dx01
        return remap_cubic(t_, t0, t1, x0, x1, dx0, dx1)

    def lagrangian01(t_, idx):
        x = linspline01(t_, ts[idx:idx+2], xs[idx:idx+2])
        y = linspline01(t_, ts[idx:idx+2], ys[idx:idx+2])
        vx = simplify(reduce_to_base(fwd_deriv(x, [(t_, 1)])))
        vy = simplify(reduce_to_base(fwd_deriv(y, [(t_, 1)])))
        vx_ = Var('vx_')
        vy_ = Var('vy_')
        kinetic01 = 1/2 * m * LetIn([vx_, vy_], [vx, vy], vx_ * vx_ + vy_ * vy_)
        potential01 = 0
        return kinetic01 - potential01
    action = Const(0)
    for idx, (t0, t1) in enumerate(zip(ts, ts[1:])):
        action += Teg(t0, t1, lagrangian01(t, idx), t)
    return action


# linear spline expression through xs; linear runtime
def condition_linspline(t: ITeg, ts: List[Union[ITeg, int]], xs: List[Union[ITeg, int]], bool_func):
    assert len(ts) == len(xs)
    assert len(ts) > 0

    f = TegFalse
    for idx, (t0, t1, x0, x1) in enumerate(zip(ts, ts[1:], xs, xs[1:])):
        indicator = t < t1 if idx == 0 else t0 <= t if idx == (len(ts) - 2) else And(t0 <= t, t < t1)
        lerp01 = remap(t, t0, t1, x0, x1)
        f = Or(f, And(indicator, bool_func(lerp01)))

    return f


def remap_cubic(t, t0, t1, x0, x1, dx0, dx1):
    u = remap(t, t0, t1, 0, 1)
    u2 = u * u
    u3 = u * u2
    return (2 * u3 - 3 * u2 + 1) * x0 + (u3 - 2 * u2 + u) * dx0 + (-2 * u3 + 3 * u2) * x1 + (u3 - u2) * dx1


def cubic_hermite_spline(t: ITeg, ts: List[Union[ITeg, int]], xs: List[Union[ITeg, int]], dxs:List[Tuple[Union[ITeg, int], Union[ITeg, int]]]):
    assert len(ts) == len(xs)
    assert len(dxs) == len(ts) - 1
    assert len(ts) > 0

    f = xs[0]
    for idx, (pt, nt, px, nx, (pdx, ndx)), in enumerate(zip(ts, ts[1:], xs, xs[1:], dxs)):
        ramp = remap_cubic(t, pt,  nt, px, nx, pdx, ndx)
        f = ramp if idx == 0 else IfElse(t < pt, f, ramp)
    return f


def solve_teg(prob: bc.BilliardsProblem, a:ITeg) -> Tuple[Optional[bc.Path], List[ITeg]]:
    start = prob.tee
    walls = prob.walls
    end = prob.pocket
    p0 = np.array([start.x, start.y])
    p1 = np.array([end.x, end.y])
    mint = start.t
    maxt = end.t

    ts = []
    us = []
    ps = []

    for idx, w in enumerate(walls):
        x0, y0 = w.x0, w.y0
        x1, y1 = w.x1, w.y1

        t = Var(f't{idx}', mint + (maxt - mint) * (idx + 1) / (len(walls) + 1))
        u = Var(f'u{idx}', 0.5)
        x = remap(u, 0, 1, x0, x1)
        y = remap(u, 0, 1, y0, y1)

        p = np.array([x, y])

        ts.append(t)
        us.append(u)
        ps.append(p)
    params = ts + us

    bound_eps = 0.01
    boundts = []
    boundus = []

    for _ in range(len(ts)):
        boundts.append((mint + bound_eps, maxt - bound_eps))
    for _ in range(len(us)):
        boundus.append((bound_eps, 1 - bound_eps))
    boundps = boundts + boundus

    lin_constraints = []
    constraints = []

    if len(ts) > 1:
        # eps <= [-1, 1, 0, 0, 0, 0] [t0,   <= np.inf
        # eps <= [0, -1, 1, 0, 0, 0]  t1,   <= np.inf
        #                             t2,
        #                             x0, x1, x2]  (assumes params pack ts first)
        lincon_lbs = np.full(len(ts) - 1, bound_eps)
        lincon_A = np.eye(len(ts) - 1, len(params), 1) - np.eye(len(ts) - 1, len(params))
        lincon_ubs = np.full_like(lincon_lbs, np.inf)
        lin_constraints.append(spop.LinearConstraint(lincon_A, lincon_lbs, lincon_ubs))

    t = TegVar('t')
    m = 1
    x = linspline(t, [mint, *ts, maxt], [p0[0], *[p[0] for p in ps], p1[0]])

    t0 = Var('t0', 5)
    t1 = Var('t1', 5)
    t2 = Var('t2', 7.5)
    x0 = Var('x0', 50)
    x1 = Var('x1', 0)
    x2 = Var('x2', 0)
    X0 = Var('X0', 0)
    X1 = Var('X1', 0)
    X2 = Var('X2', 0)
    z0 = Var('z0', 50)
    z1 = Var('z1', 0)
    z2 = Var('z2', 0)
    Z0 = Var('Z0', 0)
    Z1 = Var('Z1', 0)
    Z2 = Var('Z2', 0)
    xsd1 = Var('xsd1', 0)
    x0d0 = Var('x0d0', 0)
    x0d1 = Var('x0d1', 0)
    x1d0 = Var('x1d0', 0)
    x1d1 = Var('x1d1', 0)
    x2d0 = Var('x2d0', 0)
    x2d1 = Var('x2d1', 0)
    xfd0 = Var('xfd0', 0)
    ysd1 = Var('ysd1', 0)
    y0d0 = Var('y0d0', 0)
    y0d1 = Var('y0d1', 0)
    y1d0 = Var('y1d0', 0)
    y1d1 = Var('y1d1', 0)
    y2d0 = Var('y2d0', 0)
    y2d1 = Var('y2d1', 0)
    yfd0 = Var('yfd0', 0)
    Xsd1 = Var('Xsd1', 0)
    Xfd0 = Var('Xfd0', 0)
    zsd1 = Var('zsd1', 0)
    z0d0 = Var('z0d0', 0)
    z0d1 = Var('z0d1', 0)
    z1d0 = Var('z1d0', 0)
    z1d1 = Var('z1d1', 0)
    z2d0 = Var('z2d0', 0)
    z2d1 = Var('z2d1', 0)
    zfd0 = Var('zfd0', 0)
    Zsd1 = Var('Zsd1', 0)
    Zfd0 = Var('Zfd0', 0)
    lamb0 = Var('lamb0', 0)
    lamb1 = Var('lamb1', 0)
    svelx = Var('svelx', 0)
    svelz = Var('svelz', 0)

    def heightmap(x, z, xrange, zrange, vals):
        xlo, xhi = xrange[0], xrange[-1]
        zlo, zhi = zrange[0], zrange[-1]
        vlolo, vlohi = vals[0][0], vals[0][-1]
        vhilo, vhihi = vals[-1][0], vals[-1][-1]
        lerpxlo = remap(z, zlo, zhi, vlolo, vlohi)
        lerpxhi = remap(z, zlo, zhi, vhilo, vhihi)
        return remap(x, xlo, xhi, lerpxlo, lerpxhi)

    x = cubic_hermite_spline(t, [0, t0, 10], [0, x0, 50], [(xsd1, x0d1), (x0d1, xfd0)])
    z = cubic_hermite_spline(t, [0, t0, 10], [0, z0, 100], [(zsd1, z0d1), (z0d1, zfd0)])
    # x = linspline(t, [0, t0, t1, 10], [0, x0, x1, 50])
    # z = linspline(t, [0, t0, t1, 10], [0, z0, z1, 100])
    y = heightmap(x, z, [0, 100], [0, 100], [[0, 0], [50, 50]])
    # params = [t0, t1, x0, x1, z0, z1]
    params = [t0, x0, z0, xsd1, x0d1, xfd0, zsd1, z0d1, zfd0]
    eps = 0.1
    # boundps = [(0+eps, 10-eps), (0+eps, 10-eps), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]
    boundps = [(0+eps, 10-eps), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]
    lin_constraints = []
    # lin_constraints = [spop.LinearConstraint(np.array([-1, 1, 0, 0, 0, 0]), np.array([eps]), np.array(np.inf))]

    g = 10
    gamma = 0.4
    c = gamma

    # potential = 0
    potential = m * g * y

    vx = simplify(reduce_to_base(fwd_deriv(x, [(t, 1)])))
    vz = simplify(reduce_to_base(fwd_deriv(z, [(t, 1)])))
    vy = simplify(reduce_to_base(fwd_deriv(y, [(t, 1)])))
    # v = FwdDeriv(x, [(t, 1)]).deriv_expr
    kinetic = 1/2*m*(vx*vx + vz*vz + vy*vy)
    lagrangian = (kinetic - potential) * psmooth.Exp(gamma * t)
    # lagrangian = 1/2*m*(vx*vx + vX*vX) - c/2*(x*vX - X*vx)
    # lagrangian += 1/2*m*(vz*vz + vZ*vZ) - c/2*(z*vZ - Z*vz)
    action = Teg(mint, maxt, lagrangian, t)
    # action = cubspline_action(t, [mint, *ts, maxt],
    #                           [p0[0], *[p[0] for p in ps], p1[0]],
    #                           [p0[1], *[p[1] for p in ps], p1[1]], m)
    # forces = gamma * (vx + vz)
    # force_integral = Teg(mint, maxt, forces, t)

    timing_prev = time.time()
    print(f'started constructing: dSda')
    # dSda = RevDeriv(action, Tup(Const(1)), output_list=params)
    # dels = dSda.variables
    dSda_vars, dSda_wdelta = reverse_deriv(action, Tup(Const(1)), output_list=params)
    sdSda_wdelta = simplify(dSda_wdelta)
    saction = simplify(action)
    # sforce_integral = simplify(force_integral)
    sdSda = simplify(reduce_to_base(sdSda_wdelta))   # - sforce_integral
    dSda = sdSda
    print(f'  took: {time.time() - timing_prev}')
    print(f'dSda vars: {dSda_vars}')

    def nonzero_hessian_param_nums(pnum):
        pnum = pnum % len(ts)
        if pnum == 0:
            nonzero_tnums = [pnum, pnum + 1]
        elif pnum == len(ts) - 1:
            nonzero_tnums = [pnum - 1, pnum]
        else:
            nonzero_tnums = [pnum - 1, pnum, pnum + 1]
        return nonzero_tnums + [_ + len(ts) for _ in nonzero_tnums]

    def nonzero_hessian_params(pnum):
        nonzero_nums = nonzero_hessian_param_nums(pnum)
        return [params[nzero_num] for nzero_num in nonzero_nums]

    def expand_sparse_hessian_row(pnum, sparse_row):
        nonzero_nums = nonzero_hessian_param_nums(pnum)
        full = np.zeros(len(params))
        for nzero_num, hessval in zip(nonzero_nums, sparse_row):
            full[nzero_num] = hessval
        return full

    # hessian_sparsity = np.zeros((len(params), len(params)))
    # for param_num, row in enumerate(hessian_sparsity):
    #     for nzero_num in nonzero_hessian_param_nums(param_num):
    #         row[nzero_num] = 1

    timing_prev = time.time()
    print(f'started constructing: sdpSdas')
    # sdpSdas = [simplify(RevDeriv(saction, Tup(Const(1)), output_list=[p])) for p in params]
    _, sdpSdas = reverse_deriv(saction, Tup(Const(1)), output_list=params)
    sdpSdas = [simplify(sdpSda) for sdpSda in sdpSdas]

    print(f'started constructing: sddpSdas_sparse')

    sddpSdas_sparse = [simplify(reverse_deriv(sdpSda, Tup(Const(1)), output_list=params)[1]) for param_num, sdpSda in enumerate(sdpSdas)]
    # sddpSdas_sparse = [simplify(reverse_deriv(sdpSda, Tup(Const(1)), output_list=nonzero_hessian_params(param_num))[1]) for param_num, sdpSda in enumerate(sdpSdas)]

    # Reduce delta exprs.
    # print(f'reducing deltas in sdpSdas')
    # sdpSdas = [simplify(reduce_to_base(sdpSda)) for sdpSda in sdpSdas]

    def construct_sparse(_sddpSdas, i=None):
        # print(f'sparse {i}')
        return Tup(*(simplify(reduce_to_base(sddpSda)) for sddpSda in _sddpSdas))
    print(f'reducing deltas in sddpSdas_sparse')
    sddpSdas_sparse = [construct_sparse(_sddpSdas, i) for i, _sddpSdas in enumerate(sddpSdas_sparse)]
    # sddpSdas_sparse = [Tup(*(simplify(reduce_to_base(sddpSda)) for sddpSda in _sddpSdas)) for _sddpSdas in sddpSdas_sparse]
    print(f'  took: {time.time() - timing_prev}')

    args = Args()

    def bind_param(expr, vs):
        for param, v in zip(params, vs):
            expr.bind_variable(param, v)

    def action_func(vs):
        k = evaluate(saction, {p_: v for p_, v in zip(params, vs)}, num_samples=args.num_samples, backend=args.backend)
        # print(f'    p: {k}')
        return k

    def d_action_func(vs):
        k = evaluate(sdSda, {p_: v for p_, v in zip(params, vs)}, num_samples=args.num_samples, backend=args.backend)
        # step = 0.0001
        # vsteps = np.array([action_func(np.concatenate([vs[:i], [v + step], vs[i+1:]])) for i, v in enumerate(vs)])
        # vhere = action_func(vs)
        # k = (vsteps - vhere) / step
        # print(f'   dp: {vs}\t{k}')
        return k

    def dd_action_func(vs):
        ks = np.array([
            evaluate(sddpSda_sparse, {p_: v for p_, v in zip(params, vs)},
                     num_samples=args.num_samples, backend=args.backend)
            # expand_sparse_hessian_row(param_num_, evaluate(sddpSda_sparse, {p_: v for p_, v in zip(params, vs)},
            #                                                num_samples=args.num_samples, backend=args.backend))
            for param_num_, sddpSda_sparse in enumerate(sddpSdas_sparse)])

        # step = 0.0001
        # vsteps = np.array([action_func(np.concatenate([vs[:i], [v + step], vs[i+1:]])) for i, v in enumerate(vs)])
        # gsteps = np.array([d_action_func(np.concatenate([vs[:i], [v + step], vs[i+1:]])) for i, v in enumerate(vs)])
        # ghere = d_action_func(vs)
        # k_ = gsteps - ghere
        # k = (k_ + k_.transpose()) / (2 * step)
        # print(f'  ddp: {vs}\t{ks}')
        return ks

    timing_prev = time.time()
    print(f'compiling to c')
    action_func(np.zeros_like(params))
    d_action_func(np.zeros_like(params))
    dd_action_func(np.zeros_like(params))
    print(f'  took: {time.time() - timing_prev}')
    print()

    timing_prev = time.time()
    print(f'optimizing action')
    init_guess = np.array([p.value for p in params])
    print(f'init guess: {init_guess}')
    print(f'init   action: {action_func(init_guess)}')
    print(f'init  daction: {d_action_func(init_guess)}')
    cons = []
    for cteg in constraints:
        def cteg_func(param_vals):
            k = evaluate(cteg, {p_: v for p_, v in zip(params, param_vals)}, num_samples=args.num_samples, backend=args.backend)
            print(f'        cons: {param_vals}  =>  {k}')
            return k
        cons.append({
            'type': 'ineq',
            'fun': cteg_func
        })
    print(f'cons   :  {list(map(str, constraints))}')
    print(f'lincons: {lin_constraints}')
    res = spop.minimize(action_func, init_guess, jac=d_action_func, hess=dd_action_func, method='trust-constr', constraints=lin_constraints, bounds=boundps, options={'verbose': 2})
    # res = spop.least_squares(d_action_func, jac=dd_action_func, x0=init_guess, verbose=2)

    print(f'res: {res.x}')
    print(f'final   action: {action_func(res.x)}')
    print(f'final  daction: {d_action_func(res.x)}')
    print(f'  took: {time.time() - timing_prev}')

    tvals = []
    pvals = []
    for tval, uval, w in zip(res.x[:len(ts)], res.x[len(ts):], walls):
        x0, y0 = w.x0, w.y0
        x1, y1 = w.x1, w.y1
        xval = remap(uval, 0, 1, x0, x1)
        yval = remap(uval, 0, 1, y0, y1)
        tvals.append(tval)
        pvals.append(np.array([xval, yval]))
    path = bc.LinearPath([mint, *tvals, maxt], [p0, *pvals, p1])
    bind_param(x, res.x)
    bind_param(z, res.x)
    return path, [substitute(x, t, a), substitute(z, t, a), substitute(y, t, a)]


class Args(Tap):
    pixel_width: int = 30
    pixel_height: int = 30
    num_samples: int = 100
    t_samples: int = 100
    backend: str = 'C'  # TODO backend are different???


if __name__ == "__main__":
    w0 = bc.Wall(-15, -10, 5, -10)
    w1 = bc.Wall(5, -10, 15, 10)
    w2 = bc.Wall(15, 10, -5, 10)
    w3 = bc.Wall(-5, 10, -15, -10)

    wl = bc.Wall(-15, -15, 0, 10)
    wr = bc.Wall(-15, -15, 10, 0)
    # wl = bc.Wall(-40, 40, 10, 10)
    # wr = bc.Wall(-40, 40, -10, -10)

    start = bc.Tee(0, 0, 0)
    end = bc.Pocket(50, 100, 10)

    walls = [
        # w1,
        # w2,
        # w0,
        # w3,
        # w1,
        # w2,
        # w0,
        # w3,
        # w1,
        # w2,
        # w0,
        # w3,
        # w1,
        # w2,

        # wl,
        # wr,
        # wl,
        # wr,
        # wl,
    ]
    prob = bc.BilliardsProblem(start, walls, end)
    a = Var('a')
    path, debug = solve_teg(prob, a)
    print(f'path: {path}')
    x, z, y = debug

    args = Args()
    fig, axes = plt.subplots(nrows=2, ncols=2)

    def sample_path(p, ts_):
        for t_ in ts_:
            yield p.interpolate(t_)

    def sample_expr(expr, ts_, v=a):
        for t_ in ts_:
            f = evaluate(expr, {v: t_}, num_samples=args.num_samples, backend=args.backend)
            yield f

    ts = np.array([start.t + (end.t - start.t) * i/args.t_samples for i in range(args.t_samples + 1)])

    xs = list(sample_expr(x, ts))
    zs = list(sample_expr(z, ts))
    ys = list(sample_expr(y, ts))

    # bill_plot = axes[0]
    bill_plot = axes[0][0]
    bill_plot.plot(xs, zs)
    for w in walls:
        bill_plot.plot(np.array([w.x0, w.x1]), np.array([w.y0, w.y1]))
    bill_plot.scatter([start.x], [start.y], c='green',  marker='.')
    bill_plot.scatter([end.x], [end.y], c='red', marker='.')
    # bill_plot.set_aspect('equal', adjustable='box')

    axes[1][0].plot(ts, xs)
    axes[1][0].plot(ts, zs)
    axes[1][0].plot(ts, ys)

    # x0s = np.array([10 * i/args.t_samples for i in range(args.t_samples + 1)])

    from matplotlib import cm

    # fig = plt.figure()
    # ax = fig.add_subplot(131, projection='3d')
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax3 = fig.add_subplot(133, projection='3d')
    # # X, Y, Z = axes3d.get_test_data(0.05)
    # #
    # buff = 10

    # vals = []
    # atemp = simplify(dSda.deriv_expr)
    # for t_ in ts[::]:
    #     action.bind_variable(t0, t_)
    #     actions = np.array(list(sample_expr(atemp, x0s, v=x0)))
    #     vals.append(actions)
    #     print(f'\t{t_}')
    # vals = np.array(vals)
    # valsdt = np.diff(vals, axis=0)
    # valsdx = np.diff(vals, axis=1)
    # ax.plot_surface(np.array([x0s for _ in ts[buff:-buff-1]]), np.array([np.array([t_ for _ in x0s]) for t_ in ts[buff:-buff-1]]), valsdt[buff:-buff, :], rstride=2, cstride=1, alpha=0.5)
    # ax.contour(np.array([x0s for _ in ts[buff:-buff-1]]), np.array([np.array([t_ for _ in x0s]) for t_ in ts[buff:-buff-1]]), valsdt[buff:-buff, :], levels=[_ for _ in np.arange(-10, 6, 1)], cmap=cm.get_cmap('magma'), linestyles="solid")
    # ax.set_xlabel('x0')
    # ax.set_ylabel('t0')
    # ax.set_zlabel('dt0')
    # ax2.plot_surface(np.array([x0s[:-1] for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s[:-1]]) for t_ in ts[buff:-buff]]), valsdx[buff:-buff, :], rstride=2, cstride=1, alpha=0.5)
    # ax2.contour(np.array([x0s[:-1] for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s[:-1]]) for t_ in ts[buff:-buff]]), valsdx[buff:-buff, :], levels=[_ for _ in np.arange(-1, 2, 0.25)], cmap=cm.get_cmap('magma'), linestyles="solid")
    # ax2.set_xlabel('x0')
    # ax2.set_ylabel('t0')
    # ax2.set_zlabel('dx0')
    # ax3.plot_surface(np.array([x0s for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :], rstride=2, cstride=1, alpha=0.5)
    # ax3.contour(np.array([x0s for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :], levels=[_+12 for _ in np.arange(0, 150, 10)], cmap=cm.get_cmap('magma'), linestyles="solid")
    # ax3.set_xlabel('x0')
    # ax3.set_ylabel('t0')
    # ax3.set_zlabel('action')



    # vals = []
    # atemp = simplify(dSda.deriv_expr)
    # for t_ in ts[::]:
    #     atemp.bind_variable(t0, t_)
    #     actions = np.array(list(sample_expr(atemp, x0s, v=x0)))
    #     vals.append(actions)
    #     print(f'\t{t_}')
    # vals = np.array(vals)
    # ax.plot_surface(np.array([x0s for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :, 0], rstride=2, cstride=1, alpha=0.5)
    # ax.contour(np.array([x0s for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :, 0], levels=[10*_ for _ in np.arange(-10, 6, 1)], cmap=cm.get_cmap('magma'), linestyles="solid")
    # ax.set_xlabel('x0')
    # ax.set_ylabel('t0')
    # ax.set_zlabel('dt0')
    # ax2.plot_surface(np.array([x0s[:] for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s[:]]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :, 1], rstride=2, cstride=1, alpha=0.5)
    # ax2.contour(np.array([x0s[:] for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s[:]]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :, 1],  levels=[10*_ for _ in np.arange(-1, 2, 0.25)], cmap=cm.get_cmap('magma'), linestyles="solid")
    # ax2.set_xlabel('x0')
    # ax2.set_ylabel('t0')
    # ax2.set_zlabel('dx0')
    #
    # actions = np.array(list(sample_expr(substitute.substitute(action, t0, a), ts)))
    # dSdas = np.array(list(sample_expr(substitute.substitute(action, x0, a), xs)))
    # # dSdas = np.array(list(sample_expr(substitute.substitute(dSda.deriv_expr, t0, a), ts)))
    # axes[0][1].plot(ts, actions)
    # axes[1][1].plot(ts, dSdas)

    plt.show()


