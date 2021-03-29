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

from teg.lang.extended import (
    ITegExtended,
    Delta,
    BiMap
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

    walls = prob.walls

    mint = args.mint
    maxt = args.maxt
    ru = 1.5
    rv = 1.5

    t = TegVar('t')
    m = 1

    uts = []
    uxs = []
    uzs = []
    vts = []
    vxs = []
    vzs = []
    os = []
    utpts = []
    uxpts = []
    uzpts = []
    vtpts = []
    vxpts = []
    vzpts = []
    timewall = walls[0]
    teewall = walls[1]
    holewall = walls[2]
    teeu = np.array([teewall.x0, teewall.y0])
    teev = np.array([teewall.x1, teewall.y1])
    holeu = np.array([holewall.x0, holewall.y0])
    holev = np.array([holewall.x1, holewall.y1])
    uwall1 = walls[3]
    vwall1 = walls[4]
    vwall2 = walls[5]
    tvscale = timewall.x0

    thit1 = Var('thit1', mint + (maxt - mint)*0.2)
    tuwall1 = Var('tuwall1', mint + (maxt - mint)*0.3)
    tvwall1 = Var('tvwall1', mint + (maxt - mint)*0.3)
    thit2 = Var('thit2', mint + (maxt - mint)*0.5)
    tuwall1b = Var('tuwall1b', mint + (maxt - mint)*0.7)
    tvwall2 = Var('tvwall2', mint + (maxt - mint)*0.7)
    thit3 = Var('thit2', mint + (maxt - mint)*0.8)
    uhit1x = Var('uhit1x', teeu[0])
    uhit1z = Var('uhit1z', teeu[1])
    uwall1s = Var('uwall1s', 0.5)
    vwall1s = Var('vwall1s', 0.5)
    uhit2x = Var('uhit2x', holewall.x0 + (holewall.x1 - holewall.x0) * 0.5)
    uhit2z = Var('uhit2z', uwall1.y0 + (uwall1.y1 - uwall1.y0) * 0.5)
    uwall1bs = Var('uwall1bs', 0.5)
    vwall2s = Var('vwall2s', 0.5)
    uhit3x = Var('uhit3x', holewall.x0 + (holewall.x1 - holewall.x0) * 0.5)
    uhit3z = Var('uhit3z', uwall1.y0 + (uwall1.y1 - uwall1.y0) * 0.75)
    vhit1theta = Var('vhit1theta', 0)
    vhit2theta = Var('vhit2theta', 0)
    vhit3theta = Var('vhit3theta', 0)
    tu0 = teeu[0]
    tu1 = teeu[1]
    tu0val = teeu[0]
    tu1val = teeu[1]
    # tu0 = uhit1x
    # tu1 = uhit1z
    # tu0val = uhit1x.value
    # tu1val = uhit1z.value
    key_upoints = [
        (mint, teeu[0], teeu[1]),
        (thit1, tu0, tu1),
        (tuwall1, remap(uwall1s, 0, 1, uwall1.x0, uwall1.x1), remap(uwall1s, 0, 1, uwall1.y0, uwall1.y1)),
        (thit2, uhit2x, uhit2z),
        # (tuwall1b, remap(uwall1bs, 0, 1, uwall1.x0, uwall1.x1), remap(uwall1bs, 0, 1, uwall1.y0, uwall1.y1)),
        # (thit3, uhit3x, uhit3z),
        (maxt, holeu[0], holeu[1]),
    ]
    key_vpoints = [
        (mint, teev[0], teev[1]),
        (thit1 + 0, tu0 + (ru + rv) * smooth.Cos(vhit1theta), tu1 + (ru + rv) * smooth.Sin(vhit1theta)),
        (tvwall1, remap(vwall1s, 0, 1, vwall1.x0, vwall1.x1), remap(vwall1s, 0, 1, vwall1.y0, vwall1.y1)),
        (thit2 + 0, uhit2x + (ru + rv) * smooth.Cos(vhit2theta), uhit2z + (ru + rv) * smooth.Sin(vhit2theta)),
        # (tvwall2, remap(vwall2s, 0, 1, vwall2.x0, vwall2.x1), remap(vwall2s, 0, 1, vwall2.y0, vwall2.y1)),
        # (thit3 + 0, uhit3x + (ru + rv) * smooth.Cos(vhit3theta), uhit2z + (ru + rv) * smooth.Sin(vhit3theta)),
        (maxt*tvscale, holev[0], holev[1]),
    ]
    key_ubounds = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        # (None, None),
        # (None, None),
        (None, None),
    ]
    key_vbounds = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        # (None, None),
        # (None, None),
        (None, None),
    ]
    key_uhints = [
        (mint, teeu[0], teeu[1]),
        (thit1.value, tu0val, tu1val),
        (tuwall1.value, remap(uwall1s.value, 0, 1, uwall1.x0, uwall1.x1), remap(uwall1s.value, 0, 1, uwall1.y0, uwall1.y1)),
        (thit2.value, uhit2x.value, uhit2z.value),
        # (tuwall1b.value, remap(uwall1bs.value, 0, 1, uwall1.x0, uwall1.x1), remap(uwall1bs.value, 0, 1, uwall1.y0, uwall1.y1)),
        # (thit3.value, uhit3x.value, uhit3z.value),
        (maxt, holeu[0], holeu[1]),
    ]
    key_vhints = [
        (mint, teev[0], teev[1]),
        (thit1.value, tu0val + (ru + rv) * np.cos(vhit1theta.value), tu1val + (ru + rv) * np.sin(vhit1theta.value)),
        (tvwall1.value, remap(vwall1s.value, 0, 1, vwall1.x0, vwall1.x1), remap(vwall1s.value, 0, 1, vwall1.y0, vwall1.y1)),
        (thit2.value, uhit2x.value + (ru + rv) * np.cos(vhit2theta.value), uhit2z.value + (ru + rv) * np.sin(vhit2theta.value)),
        # (tvwall2.value, remap(vwall2s.value, 0, 1, vwall2.x0, vwall2.x1), remap(vwall2s.value, 0, 1, vwall2.y0, vwall2.y1)),
        # (thit3.value, uhit3x.value + (ru + rv) * np.cos(vhit3theta.value), uhit3z.value + (ru + rv) * np.sin(vhit3theta.value)),
        (maxt*tvscale, holev[0], holev[1]),
    ]
    k = 8
    expand_uknots = [
        0,
        k,
        k,
        k+4,
        # 0,
        k+4,
    ]
    expand_vknots = [
        k,
        k,
        k,
        k+4,
        # 0,
        k+4,
    ]
    cross_ubounds = [False for _ in key_upoints]
    cross_vbounds = [
        False,
        # False,
        True,
        False,
        # False,
        True,
        # False,
        # False,
        # # True,
        False,
    ]
    crossvaruts = [thit1, thit2]
    othervars = [
        vhit1theta,
        uwall1s,
        vwall1s,
        vhit2theta,
        # uwall1bs,
        # vwall2s,
        # vhit3theta,
    ]
    seps = 0.05
    otherbounds = [
        None,
        (seps, 1 - seps),
        (seps, 1 - seps),
        None,
        # (seps, 1 - seps),
        # (seps, 1 - seps),
        # None,
    ]

    bound_eps = args.bound_eps
    bounduts = []
    bounduxs = []
    bounduzs = []
    boundvts = []
    boundvxs = []
    boundvzs = []
    boundothers = []
    crossvaridxs = []
    def add_othervars(othervars_, otherbounds_):
        os.extend(othervars_)
        boundothers.extend([b if b is not None else (-np.inf, np.inf) for b in otherbounds_])
    add_othervars(othervars, otherbounds)
    def expand_key_knots(key_points, key_bounds, key_hints, expand_knots, cross_bounds, tvars, xvars, zvars, tpts_, xpts, zpts, tbounds, xbounds, zbounds, keyname='u', tscale=1.0):
        for idx, (k0, k1, k0_bounds, k0_hint, k1_hint, numknots, cross_b) in enumerate(zip(key_points, key_points[1:], key_bounds, key_hints, key_hints[1:], expand_knots, cross_bounds)):
            k0t, k0x, k0z = k0
            k0bx, k0bz = k0_bounds
            tpts_.append(k0t)
            xpts.append(k0x)
            zpts.append(k0z)
            bx = (-np.inf, np.inf)
            bz = (-np.inf, np.inf)
            if cross_b:
                prevtidx = len(tvars)-1
                crossvaridxs.append([prevtidx, prevtidx+1])
            if isinstance(k0t, Var):
                tvars.append(k0t)
                tbounds.append((mint+bound_eps, maxt*tscale-bound_eps))
            if isinstance(k0x, Var):
                xvars.append(k0x)
                if k0bx is not None:
                    if k0bx[0] is None:
                        bx = k0bx[1]
                    else:
                        bx = k0bx
                xbounds.append(bx)
            if isinstance(k0z, Var):
                zvars.append(k0z)
                if k0bz is not None:
                    if k0bz[0] is None:
                        bz = k0bz[1]
                    else:
                        bz = k0bz
                zbounds.append(bz)
            for i in range(numknots):
                tvar = Var(f't_{keyname}_{idx}_{i}', remap(i+1, 0, numknots+1, k0_hint[0], k1_hint[0]))
                xvar = Var(f'x_{keyname}_{idx}_{i}', remap(i+1, 0, numknots+1, k0_hint[1], k1_hint[1]))
                zvar = Var(f'z_{keyname}_{idx}_{i}', remap(i+1, 0, numknots+1, k0_hint[2], k1_hint[2]))
                tvars.append(tvar)
                xvars.append(xvar)
                zvars.append(zvar)
                tpts_.append(tvar)
                xpts.append(xvar)
                zpts.append(zvar)
                tbounds.append((mint, maxt*tscale))
                # if k0_yonramp:
                #     boundxs.append((-np.inf, modrampx))
                # else:
                xbounds.append(bx)
                zbounds.append(bz)
        tpts_.append(key_points[-1][0])
        xpts.append(key_points[-1][1])
        zpts.append(key_points[-1][2])

    expand_key_knots(key_upoints, key_ubounds, key_uhints, expand_uknots, cross_ubounds, uts, uxs, uzs, utpts, uxpts, uzpts, bounduts, bounduxs, bounduzs, 'u')
    expand_key_knots(key_vpoints, key_vbounds, key_vhints, expand_vknots, cross_vbounds, vts, vxs, vzs, vtpts, vxpts, vzpts, boundvts, boundvxs, boundvzs, 'v', tscale=tvscale)

    ux = linspline(t, utpts, uxpts)
    uz = linspline(t, utpts, uzpts)
    vx = linspline(t, vtpts, vxpts)
    vz = linspline(t, vtpts, vzpts)

    params = uts + vts + uxs + uzs + vxs + vzs + os
    boundps = bounduts + boundvts + bounduxs + bounduzs + boundvxs + boundvzs + boundothers
    crossvarbounds = []
    for (lidx, uidx), uvar in zip(crossvaridxs, crossvaruts):
        uvaridx = uts.index(uvar)
        crossvarbounds.append([len(uts)+lidx, uvaridx])
        crossvarbounds.append([uvaridx, len(uts)+uidx])
    bound_eps = args.bound_eps
    bound_scale = args.bound_scale
    lin_constraints = []
    def add_all_bounds(tsets, crossvarbounds):
        lincon_lbs_ = []
        lincon_As = []
        lincon_ubs_ = []
        def add_bounds(ts, num_params_before_):
            if len(ts) <= 1:
                return
            # eps <= [-1, 1, 0, 0, 0, 0] [t0,   <= np.inf
            # eps <= [0, -1, 1, 0, 0, 0]  t1,   <= np.inf
            #                             t2,
            #                             x0, x1, x2]  (assumes params pack ts first)
            lbs = np.full(len(ts) - 1, bound_eps)
            A = np.eye(len(ts) - 1, len(params), 1 + num_params_before_) - np.eye(len(ts) - 1, len(params), num_params_before_)
            ubs = np.full_like(lbs, np.inf)
            lincon_lbs_.extend(lbs)
            lincon_As.append(A)
            lincon_ubs_.extend(ubs)
        num_params_before = 0
        for ts_ in tsets:
            add_bounds(ts_, num_params_before)
            num_params_before += len(ts_)
        for lvaridx, uvaridx, in crossvarbounds:
            lincon_lbs_.append(bound_eps)
            Avec = np.zeros(len(params))
            Avec[lvaridx] = -1
            Avec[uvaridx] = 1
            lincon_As.append(np.array([Avec]))
            lincon_ubs_.append(np.inf)
        if len(lincon_lbs_) > 0:
            lincon_lbs = np.array(lincon_lbs_)
            lincon_A = np.vstack(lincon_As)
            lincon_ubs = np.array(lincon_ubs_)
            lin_constraints.append(spop.LinearConstraint(lincon_A*bound_scale, lincon_lbs, lincon_ubs))

    add_all_bounds([uts, vts], crossvarbounds)
    g = 10
    gamma = args.gamma
    c = gamma

    def mod_linspline_action(t, ts, xs, zs):
        def linspline01(t_, t01, x01):
            t0, t1 = t01
            x0, x1 = x01
            return remap(t_, t0, t1, x0, x1)

        def lagrangian01(t_, idx):
            x = linspline01(t_, ts[idx:idx + 2], xs[idx:idx + 2])
            z = linspline01(t_, ts[idx:idx + 2], zs[idx:idx + 2])
            vx = simplify(reduce_to_base(fwd_deriv(x, [(t_, 1)])))
            vz = simplify(reduce_to_base(fwd_deriv(z, [(t_, 1)])))
            kinetic01 = 1/2 * m * (smooth.Sqr(vx) + smooth.Sqr(vz))
            return kinetic01 * psmooth.Exp(gamma / m * t_)

        action = Const(0)
        for idx, (t0, t1) in enumerate(zip(ts, ts[1:])):
            action += Teg(t0, t1, lagrangian01(t, idx), t)
        return action

    vux = simplify(reduce_to_base(fwd_deriv(ux, [(t, 1)])))
    vuz = simplify(reduce_to_base(fwd_deriv(uz, [(t, 1)])))
    vvx = simplify(reduce_to_base(fwd_deriv(vx, [(t, 1)])))
    vvz = simplify(reduce_to_base(fwd_deriv(vz, [(t, 1)])))
    # v = FwdDeriv(x, [(t, 1)]).deriv_expr
    kineticu = 1/2*m*(smooth.Sqr(vux) + smooth.Sqr(vuz))
    kineticv = 1/2*m*(smooth.Sqr(vvx) + smooth.Sqr(vvz))
    lagrangian = (kineticu + kineticv) * psmooth.Exp(gamma / m * t)
    # action = Teg(mint, maxt, lagrangian, t)
    action = mod_linspline_action(t=t, ts=utpts, xs=uxpts, zs=uzpts) + mod_linspline_action(t=t, ts=vtpts, xs=vxpts, zs=vzpts)

    # forces = gamma * (vx + vz)
    # force_integral = Teg(mint, maxt, forces, t)

    timing_prev = time.time()
    print(f'started constructing: dSda')
    # dSda = RevDeriv(action, Tup(Const(1)), output_list=params)
    # dels = dSda.variables

    dSda_vars, dSda_wdelta = reverse_deriv(action, Tup(Const(1)), output_list=params, args={'ignore_deltas': True, 'ignore_bounds': True})
    sdSda_wdelta = simplify(dSda_wdelta)
    saction = simplify(action)
    # sforce_integral = simplify(force_integral)
    sdSda = simplify(sdSda_wdelta)   # - sforce_integral
    dSda = sdSda
    print(f'  took: {time.time() - timing_prev}')
    print(f'dSda vars: {dSda_vars}')
    # def nonzero_hessian_param_nums(pnum):
    #     sparsity = np.array([
    #         [1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
    #     ])
    #     pnum = pnum % len(ts)
    #     if pnum == 0:
    #         nonzero_tnums = [pnum, pnum + 1]
    #     elif pnum == len(ts) - 1:
    #         nonzero_tnums = [pnum - 1, pnum]
    #     else:
    #         nonzero_tnums = [pnum - 1, pnum, pnum + 1]
    #     return nonzero_tnums + [_ + len(ts) for _ in nonzero_tnums]
    #     # return sparsity[pnum]
    #
    # def nonzero_hessian_params(pnum):
    #     nonzero_nums = nonzero_hessian_param_nums(pnum)
    #     return [params[nzero_num] for nzero_num in nonzero_nums]
    #
    # def expand_sparse_hessian_row(pnum, sparse_row):
    #     nonzero_nums = nonzero_hessian_param_nums(pnum)
    #     full = np.zeros(len(params))
    #     for nzero_num, hessval in zip(nonzero_nums, sparse_row):
    #         full[nzero_num] = hessval
    #     return full

    # hessian_sparsity = np.zeros((len(params), len(params)))
    # for param_num, row in enumerate(hessian_sparsity):
    #     for nzero_num in nonzero_hessian_param_nums(param_num):
    #         row[nzero_num] = 1

    timing_prev = time.time()
    print(f'started constructing: sdpSdas')
    # sdpSdas = [simplify(RevDeriv(saction, Tup(Const(1)), output_list=[p])) for p in params]
    _, sdpSdas = reverse_deriv(saction, Tup(Const(1)), output_list=params, args={'ignore_deltas': True, 'ignore_bounds': True})
    sdpSdas = [simplify(sdpSda) for sdpSda in sdpSdas]

    print(f'started constructing: sddpSdas_sparse')

    sddpSdas_sparse = [simplify(reverse_deriv(sdpSda, Tup(Const(1)), output_list=params, args={'ignore_deltas': True, 'ignore_bounds': True})[1]) for param_num, sdpSda in enumerate(sdpSdas)]
    # sddpSdas_sparse = [simplify(reverse_deriv(sdpSda, Tup(Const(1)), output_list=nonzero_hessian_params(param_num))[1]) for param_num, sdpSda in enumerate(sdpSdas)]

    # Reduce delta exprs.
    # print(f'reducing deltas in sdpSdas')
    # sdpSdas = [simplify(reduce_to_base(sdpSda)) for sdpSda in sdpSdas]

    def construct_sparse(_sddpSdas, i=None):
        # print(f'sparse {i}')
        return Tup(*(sddpSda for sddpSda in _sddpSdas))
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
    constraints = []
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

    # class Res(object):
    #     def __init__(self, x):
    #         self.x = x
    # res = Res(init_guess)
    print(f'res: {res.x}')
    print(f'final   action: {action_func(res.x)}')
    print(f'final  daction: {d_action_func(res.x)}')
    print(f'  took: {time.time() - timing_prev}')

    bind_param(ux, res.x)
    bind_param(uz, res.x)
    bind_param(vx, res.x)
    bind_param(vz, res.x)
    bind_param(lagrangian, res.x)
    path = bc.LinearPath([0, 0], [0, 0])
    return path, [substitute(ux, t, a), substitute(uz, t, a), substitute(vx, t, a), substitute(vz, t, a), substitute(lagrangian, t, a)]


class Args(Tap):
    num_samples: int = 400
    t_samples: int = 100
    backend: str = 'C'

    gamma: float = 0.2
    bound_eps: float = 0.3
    bound_scale: float = 100000000
    mint: float = 0
    maxt: float = 15


def main():
    args = Args()

    tvscale = 1.2
    timewall = bc.Wall(tvscale, 0, 0, 0)
    teeu = np.array([9.3, -14.4])
    # teeu = np.array([5, 0])
    teev = np.array([12, -20])
    holeu = np.array([-2, 24])
    holev = np.array([30, 20])
    teewall = bc.Wall(teeu[0], teeu[1], teev[0], teev[1])
    holewall = bc.Wall(holeu[0], holeu[1], holev[0], holev[1])
    uwall1 = bc.Wall(4, -20, holeu[0], holeu[1])
    vwall1 = bc.Wall(22, -17, 26, 2.3)
    vwall2 = bc.Wall(26, 2.3, holev[0], holev[1])
    # vwall2 = bc.Wall(vwall1.x1, vwall1.y1, holev[0], holev[1])

    scalefactor = 1
    start = bc.Tee(14.337 * scalefactor, 17.441 * scalefactor, args.mint)
    end = bc.Pocket(23.302 * scalefactor, 0.354 * scalefactor, args.maxt)

    walls = [
        timewall,
        teewall,
        holewall,
        uwall1,
        vwall1,
        vwall2,
    ]
    prob = bc.BilliardsProblem(start, walls, end)
    a = Var('a')
    path, debug = solve_teg(prob, a)
    print(f'path: {path}')
    ux, uz, vx, vz, lagrangian = debug

    fig, axes = plt.subplots(nrows=3, ncols=2)

    def sample_path(p, ts_):
        for t_ in ts_:
            yield p.interpolate(t_)

    def sample_expr(expr, ts_, v=a):
        for t_ in ts_:
            f = evaluate(expr, {v: t_}, num_samples=args.num_samples, backend=args.backend)
            yield f

    ts = np.array([start.t + (end.t - start.t) * i/args.t_samples for i in range(args.t_samples + 1)])
    tsv = np.array([start.t + (end.t*tvscale - start.t) * i/args.t_samples for i in range(args.t_samples + 1)])
    ts2 = np.array([start.t + i/24 for i in range(24*(end.t-start.t) + 1)])
    tsv2 = np.array([start.t + i/24 for i in range(int(24*(end.t*tvscale-start.t) + 1))])

    uxs = list(sample_expr(ux, ts))
    uzs = list(sample_expr(uz, ts))
    vxs = list(sample_expr(vx, tsv))
    vzs = list(sample_expr(vz, tsv))
    duxs = np.array(list(sample_expr(FwdDeriv(ux, [(a, 1)]), ts)))
    duzs = np.array(list(sample_expr(FwdDeriv(uz, [(a, 1)]), ts)))
    dvxs = np.array(list(sample_expr(FwdDeriv(vx, [(a, 1)]), ts)))
    dvzs = np.array(list(sample_expr(FwdDeriv(vz, [(a, 1)]), ts)))
    # ls = list(sample_expr(lagrangian, ts))
    uxs2 = np.array(list(sample_expr(ux, ts2)))
    uzs2 = np.array(list(sample_expr(uz, ts2)))
    vxs2 = np.array(list(sample_expr(vx, tsv2)))
    vzs2 = np.array(list(sample_expr(vz, tsv2)))
    txyzs = np.array([(idx, t, (ux, uz)) for idx, (t, ux, uz) in enumerate(zip(ts2, uxs2, uzs2))])
    print(f'positions = [')
    for _ in txyzs:
        print(f'  {_},')
    print(f']')
    txyzs = np.array([(idx, t, (vx, vz)) for idx, (t, vx, vz) in enumerate(zip(tsv2, vxs2, vzs2))])
    print(f'positions = [')
    for _ in txyzs:
        print(f'  {_},')
    print(f']')

    # bill_plot = axes[0]
    bill_plot = axes[0][0]
    bill_plot.plot(uxs, uzs)
    bill_plot.plot(vxs, vzs)
    for w in walls[2:]:
        bill_plot.plot(np.array([w.x0, w.x1]), np.array([w.y0, w.y1]))
    bill_plot.scatter([teeu[0], holeu[0]], [teeu[1], holeu[1]], c='green',  marker='.')
    bill_plot.scatter([teev[0], holeu[0]], [teev[1], holeu[1]], c='red', marker='.')
    # bill_plot.set_aspect('equal', adjustable='box')

    axes[1][0].plot(ts, uxs)
    axes[1][0].plot(ts, uzs)
    axes[1][0].plot(tsv, vxs)
    axes[1][0].plot(tsv, vzs)

    axes[2][0].plot(ts, duxs)
    axes[2][0].plot(ts, dvxs)
    axes[2][0].plot(ts, duxs + dvxs)
    axes[2][1].plot(ts, duzs)
    axes[2][1].plot(ts, dvzs)
    axes[2][1].plot(ts, duzs + dvzs)

    axes[1][1].plot(ts, duxs*duxs + duzs*duzs)
    axes[1][1].plot(ts, dvxs*dvxs + dvzs*dvzs)
    axes[1][1].plot(ts, duxs*duxs + duzs*duzs + dvxs*dvxs + dvzs*dvzs)


    plt.show()
    k = 1


if __name__ == "__main__":
    main()
