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
    args = Args()
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

    bound_eps = 0.00
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

    def heightmap(x, z, xrange, zrange, vals):
        xlo, xhi = xrange[0], xrange[-1]
        zlo, zhi = zrange[0], zrange[-1]
        vlolo, vlohi = vals[0][0], vals[0][-1]
        vhilo, vhihi = vals[-1][0], vals[-1][-1]
        lerpxlo = remap(z, zlo, zhi, vlolo, vlohi)
        lerpxhi = remap(z, zlo, zhi, vhilo, vhihi)
        return remap(x, xlo, xhi, lerpxlo, lerpxhi)

    scalefactor = 1
    tee = np.array([14.337, 17.441]) * scalefactor
    hole = np.array([23.302,  0.354]) * scalefactor
    rampx = -10.605 * scalefactor
    ballr = 0.253 * scalefactor
    # modrampx = rampx + ballr * 2
    modrampx = rampx
    A = np.array([-10.605, 20.389]) * scalefactor
    B = np.array([12.727, 26.91]) * scalefactor
    C = np.array([26.504, 13.185]) * scalefactor
    D = np.array([1.056, 5.692]) * scalefactor
    E = np.array([1.351, 4.728]) * scalefactor
    F = np.array([26.783, 12.211]) * scalefactor
    G = np.array([26.783, -10.815]) * scalefactor
    H = np.array([0.835, -4.655]) * scalefactor
    S = np.array([-2.502, 1.633]) * scalefactor
    T = np.array([-2.502, -1.633]) * scalefactor
    U = np.array([2.525, 1.633]) * scalefactor
    V = np.array([2.525, -1.633]) * scalefactor
    mint = args.mint
    maxt = args.maxt

    ton = Var('ton', mint + (maxt-mint)*0.2)
    tup = Var('tup', mint + (maxt-mint)*0.5)
    toff = Var('toff', mint + (maxt-mint)*0.7)
    tst = Var('tst', mint + (maxt-mint)*0.7)
    tuv = Var('tuv', mint + (maxt-mint)*0.8)
    xon = Var('xon', modrampx - 5)
    xup = Var('xup', modrampx - 10)
    xoff = Var('xoff', modrampx - 5)
    zon = Var('zon', A[1])
    zup = Var('zup', D[1])
    zoff = Var('zoff', H[1])
    zst = Var('zst', (S[1] + T[1])/2)
    zuv = Var('zuv', (U[1] + V[1])/2)

    ts = []
    xs = []
    zs = []
    tpts = []
    xpts = []
    zpts = []
    # start_key = (mint, tee[0], tee[1])
    # key_points = [(ton, modrampx, zon), (toff, modrampx, zoff), (tst, S[0], zst), (tuv, U[0], zuv)]
    # key_bounds = [(None, None), ((modrampx, modrampx), None), ((modrampx, modrampx), None), ((S[0], S[0]), (T[1], S[1])), ((U[0], U[0]), (V[1], U[1]))]
    # end_key = (maxt, hole[0], hole[1])
    # key_hints = [(mint, tee[0], tee[1]), (2, modrampx - 5, (A[1] + D[1])/2), (5, modrampx - 5, (S[1] + T[1])/2), (7, S[0], (S[1] + T[1])/2), (8, U[0], (S[1] + T[1])/2), (10, hole[0], hole[1])]
    # expand_knots = [3, 8, 2, 4, 8]
    # key_yonramps = [False, True, False, False, False]
    key_points = [
        (mint, tee[0], tee[1]),
        (ton, xon, zon),
        # (tup, xup, zup),
        (toff, xoff, zoff),
        # (tst, S[0], zst),
        # (tuv, U[0], zuv),
        (maxt, hole[0], hole[1]),
    ]
    key_bounds = [
        (None, None),
        # ((None, (modrampx-50, 0)), (D[1], np.inf)),
        (None, None),
        (None, None),
        # ((None, (modrampx-50, 0)), (-np.inf, D[1])),
        # (None, None),
        # (None, (T[1], S[1])),
        # (None, (V[1], U[1])),
        (None, None),
    ]
    key_hints = [
        (mint, tee[0], tee[1]),
        (ton.value, xon.value, zon.value),
        # (tup.value, xup.value, zup.value),
        (toff.value, xoff.value, zoff.value),
        # (tst.value, S[0], zst.value),
        # (tuv.value, U[0], zuv.value),
        (maxt, hole[0], hole[1]),
    ]
    expand_knots = [
        8,
        16,
        8,
    ]
    key_yonramps = [True for _ in expand_knots]

    bound_eps = args.bound_eps
    boundts = []
    boundxs = []
    boundzs = []
    yonramps = []
    for idx, (k0, k1, k0_bounds, k0_hint, k1_hint, k0_yonramp, numknots) in enumerate(zip(key_points, key_points[1:], key_bounds, key_hints, key_hints[1:], key_yonramps, expand_knots)):
        k0t, k0x, k0z = k0
        k0bx, k0bz = k0_bounds
        yonramps.append(k0_yonramp)
        tpts.append(k0t)
        xpts.append(k0x)
        zpts.append(k0z)
        bx = (-np.inf, np.inf)
        bz = (-np.inf, np.inf)
        if isinstance(k0t, Var):
            ts.append(k0t)
            boundts.append((mint+bound_eps, maxt-bound_eps))
        if isinstance(k0x, Var):
            xs.append(k0x)
            if k0bx is not None:
                if k0bx[0] is None:
                    bx = k0bx[1]
                else:
                    bx = k0bx
            boundxs.append(bx)
        if isinstance(k0z, Var):
            zs.append(k0z)
            if k0bz is not None:
                if k0bz[0] is None:
                    bz = k0bz[1]
                else:
                    bz = k0bz
            boundzs.append(bz)
        for i in range(numknots):
            tvar = Var(f't{idx}_{i}', remap(i+1, 0, numknots+1, k0_hint[0], k1_hint[0]))
            xvar = Var(f'x{idx}_{i}', remap(i+1, 0, numknots+1, k0_hint[1], k1_hint[1]))
            zvar = Var(f'z{idx}_{i}', remap(i+1, 0, numknots+1, k0_hint[2], k1_hint[2]))
            ts.append(tvar)
            xs.append(xvar)
            zs.append(zvar)
            tpts.append(tvar)
            xpts.append(xvar)
            zpts.append(zvar)
            boundts.append((mint, maxt))
            # if k0_yonramp:
            #     boundxs.append((-np.inf, modrampx))
            # else:
            boundxs.append(bx)
            boundzs.append(bz)
            yonramps.append(k0_yonramp)
    tpts.append(key_points[-1][0])
    xpts.append(key_points[-1][1])
    zpts.append(key_points[-1][2])

    x = linspline(t, tpts, xpts)
    z = linspline(t, tpts, zpts)
    def gauss(a, b, c, x):
        return a * psmooth.Exp(-0.5 * smooth.Sqr(x-b) / (c*c))
    def yfunc(x):
        return gauss(30, modrampx - 27, 12, x)
        # return gauss(15, modrampx - 25, 10, x)
        # return -x
    y = yfunc(x)
    # y = Const(0)
    params = ts + xs + zs
    boundps = boundts + boundxs + boundzs
    bound_eps = args.bound_eps
    bound_scale = args.bound_scale
    # boundps = ([(mint+eps, maxt-eps) for _ in ts] + [(-np.inf, modrampx), (-np.inf, modrampx)] +
    #            [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (T[1], S[1]), (V[1], U[1])])
    lin_constraints = []
    if len(ts) > 1:
        # eps <= [-1, 1, 0, 0, 0, 0] [t0,   <= np.inf
        # eps <= [0, -1, 1, 0, 0, 0]  t1,   <= np.inf
        #                             t2,
        #                             x0, x1, x2]  (assumes params pack ts first)
        lincon_lbs = np.full(len(ts) - 1, bound_eps)
        lincon_A = np.eye(len(ts) - 1, len(params), 1) - np.eye(len(ts) - 1, len(params))
        lincon_ubs = np.full_like(lincon_lbs, np.inf)
        lin_constraints.append(spop.LinearConstraint(lincon_A*bound_scale, lincon_lbs, lincon_ubs))

    g = 10
    gamma = args.gamma
    c = gamma

    def mod_linspline_action(t, ts, xs, zs, yonramps):
        def linspline01(t_, t01, x01):
            t0, t1 = t01
            x0, x1 = x01
            return remap(t_, t0, t1, x0, x1)

        def lagrangian01(t_, idx):
            x = linspline01(t_, ts[idx:idx + 2], xs[idx:idx + 2])
            z = linspline01(t_, ts[idx:idx + 2], zs[idx:idx + 2])
            y = yfunc(x)
            vx = simplify(reduce_to_base(fwd_deriv(x, [(t_, 1)])))
            vz = simplify(reduce_to_base(fwd_deriv(z, [(t_, 1)])))
            vy = simplify(reduce_to_base(fwd_deriv(y, [(t_, 1)])))
            # vx_ = Var('vx_')
            # vz_ = Var('vz_')
            # vy_ = Var('vy_')
            # kinetic01 = 1 / 2 * m * LetIn([vx_, vz_, vy_], [vx, vz, vy], vx_ * vx_ + vz_ * vz_ + vy_ * vy_)
            kinetic01 = 1 / 2 * m * (vx * vx + vz * vz + vy * vy)
            potential01 = m * g * y
            return (kinetic01 - potential01) * psmooth.Exp(gamma / m * t_)

        action = Const(0)
        for idx, (t0, t1) in enumerate(zip(ts, ts[1:])):
            action += Teg(t0, t1, lagrangian01(t, idx), t)
        return action

    # # potential = 0
    potential = m * g * y
    #
    vx = simplify(reduce_to_base(fwd_deriv(x, [(t, 1)])))
    vz = simplify(reduce_to_base(fwd_deriv(z, [(t, 1)])))
    vy = simplify(reduce_to_base(fwd_deriv(y, [(t, 1)])))
    # v = FwdDeriv(x, [(t, 1)]).deriv_expr
    kinetic = 1/2*m*(vx*vx + vz*vz + vy*vy)
    lagrangian = (kinetic - potential) * psmooth.Exp(gamma * t)
    # action = Teg(mint, maxt, lagrangian, t)
    action = mod_linspline_action(t=t, ts=tpts, xs=xpts, zs=zpts, yonramps=yonramps)

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
        sparsity = np.array([
            [1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        ])
        pnum = pnum % len(ts)
        if pnum == 0:
            nonzero_tnums = [pnum, pnum + 1]
        elif pnum == len(ts) - 1:
            nonzero_tnums = [pnum - 1, pnum]
        else:
            nonzero_tnums = [pnum - 1, pnum, pnum + 1]
        return nonzero_tnums + [_ + len(ts) for _ in nonzero_tnums]
        # return sparsity[pnum]

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
        # print(f'  ddp')
        # for s in ks:
        #     print(f'    {s}')
        # print()

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
    res = spop.minimize(action_func, init_guess, jac=d_action_func, hess=dd_action_func, method='trust-constr', constraints=lin_constraints, bounds=boundps, options={'maxiter': 1000, 'verbose': 2})
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
    return path, [substitute(x, t, a), substitute(z, t, a), substitute(y, t, a), substitute(lagrangian, t, a)]


class Args(Tap):
    num_samples: int = 400
    t_samples: int = 500
    backend: str = 'C'

    gamma: float = 0.2
    bound_eps: float = 0.2
    bound_scale: float = 100000000
    mint: float = 0
    maxt: float = 10


if __name__ == "__main__":
    args = Args()
    w0 = bc.Wall(-15, -10, 5, -10)
    w1 = bc.Wall(5, -10, 15, 10)
    w2 = bc.Wall(15, 10, -5, 10)
    w3 = bc.Wall(-5, 10, -15, -10)

    wl = bc.Wall(-15, -15, 0, 10)
    wr = bc.Wall(-15, -15, 10, 0)
    # wl = bc.Wall(-40, 40, 10, 10)
    # wr = bc.Wall(-40, 40, -10, -10)

    scalefactor = 1
    start = bc.Tee(14.337 * scalefactor, 17.441 * scalefactor, args.mint)
    end = bc.Pocket(23.302 * scalefactor, 0.354 * scalefactor, args.maxt)
    A = np.array([-10.605, 20.389]) * scalefactor
    S = np.array([-2.502, 1.633]) * scalefactor
    T = np.array([-2.502, -1.633]) * scalefactor
    U = np.array([2.525, 1.633]) * scalefactor
    V = np.array([2.525, -1.633]) * scalefactor
    wsu = bc.Wall(S[0], S[1], U[0], U[1])
    wtv = bc.Wall(T[0], T[1], V[0], V[1])
    rampx = -10.605 * scalefactor
    ballr = 0.253 * scalefactor
    modrampx = rampx + ballr * 2
    wramp = bc.Wall(modrampx, A[1], modrampx, T[1])

    walls = [
        wsu,
        wtv,
        wramp,
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
    x, z, y, lagrangian = debug

    fig, axes = plt.subplots(nrows=2, ncols=2)

    def sample_path(p, ts_):
        for t_ in ts_:
            yield p.interpolate(t_)

    def sample_expr(expr, ts_, v=a):
        for t_ in ts_:
            f = evaluate(expr, {v: t_}, num_samples=args.num_samples, backend=args.backend)
            yield f

    ts = np.array([start.t + (end.t - start.t) * i/args.t_samples for i in range(args.t_samples + 1)])
    ts2 = np.array([start.t + i/24 for i in range(24*(end.t-start.t) + 1)])

    xs = list(sample_expr(x, ts))
    zs = list(sample_expr(z, ts))
    ys = list(sample_expr(y, ts))
    # dxs = list(sample_expr(FwdDeriv(x, [(a, 1)]), ts))
    # dzs = list(sample_expr(FwdDeriv(z, [(a, 1)]), ts))
    # dys = list(sample_expr(FwdDeriv(y, [(a, 1)]), ts))
    # ls = list(sample_expr(lagrangian, ts))
    xs2 = np.array(list(sample_expr(x, ts2)))
    zs2 = np.array(list(sample_expr(z, ts2)))
    ys2 = np.array(list(sample_expr(y, ts2)))
    print(f'ts2:')
    print(f'  {ts2}')
    print(f'xs2:')
    print(f'  {xs2}')
    print(f'zs2:')
    print(f'  {zs2}')
    print(f'ys2:')
    print(f'  {ys2}')
    txyzs = np.array([f'{(idx, t, (x, y, z))},' for idx, (t, x, y, z) in enumerate(zip(ts2, xs2, ys2, zs2))])
    print(f't,x,y,z::')
    print(f'  {txyzs}')

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

    # axes[1][1].plot(ts, dxs)
    # axes[1][1].plot(ts, dzs)
    # axes[1][1].plot(ts, dys)

    # axes[0][1].plot(ts, ls)

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


