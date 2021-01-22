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
from teg.derivs import FwdDeriv, RevDeriv
from teg.derivs.reverse_deriv import reverse_deriv
from teg.derivs.fwd_deriv import fwd_deriv
from teg.eval import evaluate
from teg.passes import simplify
from teg.passes import substitute
from tap import Tap
from teg.passes.simplify import simplify
from teg.passes.reduce import reduce_to_base

def remap(t, t0, t1, x0, x1):
    s = (t - t0) / (t1 - t0)
    return x0 + s*(x1 - x0)


def linspline(t: ITeg, ts: List[ITeg], xs: List[ITeg]):
    assert len(ts) == len(xs)
    assert len(ts) > 0

    f = Const(0)
    for idx, (t0, t1, x0, x1) in enumerate(zip(ts, ts[1:], xs, xs[1:])):
        indicator = t < t1 if idx == 0 else t0 <= t if idx == (len(ts) - 2) else And(t0 <= t, t < t1)
        lerp01 = remap(t, t0, t1, x0, x1)
        f += IfElse(indicator, lerp01, Const(0))
    return f


def linspline_action(t: ITeg, ts: List[ITeg], xs: List[ITeg], ys: List[ITeg], m: Union[ITeg, int]):
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
def condition_linspline(t: ITeg, ts: List[ITeg], xs: List[ITeg], bool_func):
    assert len(ts) == len(xs)
    assert len(ts) > 0

    f = TegFalse
    for idx, (t0, t1, x0, x1) in enumerate(zip(ts, ts[1:], xs, xs[1:])):
        indicator = t < t1 if idx == 0 else t0 <= t if idx == (len(ts) - 2) else And(t0 <= t, t < t1)
        lerp01 = remap(t, t0, t1, x0, x1)
        f = Or(f, And(indicator, bool_func(lerp01)))

    return f


def test_func(t, a, m):
    t_ = TegVar('t_')

    # a_ = a
    # a = t
    params = [a]

    ts = [0, 1, 2]
    xs = [0, a, 0]
    x = linspline(t_, ts, xs)
    x_cond = condition_linspline(t_, ts, xs, lambda expr: (1 < expr))
    k = 10000
    potential = k * IfElse(x_cond, smooth.Sqr(x - 1), 0)
    # potential = k * IfElse(x_cond, 1, 0)

    v = simplify(FwdDeriv(x, [(t_, 1), *[(_, 0) for _ in params]]).deriv_expr)
    kinetic = 1/2*m*v*v
    lagrangian = kinetic - potential
    action = Teg(0, 2, lagrangian, t_)
    #dSda = RevDeriv(action, Tup(Const(1)))
    dSda_vars, dSda = reverse_deriv(action, Tup(Const(1)))
    dSda = simplify(dSda)
    print(dSda_vars)

    args = Args()

    def bind_param(expr, vs):
        for param, v in zip(params, vs):
            expr.bind_variable(param, v)

    def action_func(vs):
        bind_param(action, vs)
        return evaluate(action, num_samples=args.num_samples, ignore_cache=True)

    def d_action_func(vs):
        bind_param(dSda, vs)
        k = evaluate(dSda, num_samples=args.num_samples, ignore_cache=True)
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


def convert_to_teg(prob: bc.BilliardsProblem):
    start = prob.tee
    walls = prob.walls
    end = prob.pocket
    ts = [Var(f't{idx}', (idx + 1) / (len(walls) + 1)) for idx, w in enumerate(walls)]
    xs = [Var(f'x{idx}', (w.x0 + w.x1) / 2) for idx, w in enumerate(walls)]
    ys = [Var(f'y{idx}', (w.y0 + w.y1) / 2) for idx, w in enumerate(walls)]
    params = ts + xs + ys
    collision_params = [Tup(x, y) for x, y in zip(xs, ys)]
    ts = [0, *ts, 1]
    ps = [Tup(start.x, start.y), *collision_params, Tup(end.x, end.y)]
    t = TegVar('t')
    m = Var('m', 1)

    x = linspline(t, ts, ps)
    # x_cond = condition_linspline(t, ts, ps, lambda expr: (1 < expr))

    potential = 0

    # v = FwdDeriv(x, [(t, 1), *[(_, 0) for _ in params]]).deriv_expr
    v = simplify(reduce_to_base(fwd_deriv(x, [(t, 1), *[(_, 0) for _ in params]])))
    kinetic = 1/2*m*v*v
    lagrangian = kinetic - potential
    action = Teg(0, 1, lagrangian, t)
    #dSda = RevDeriv(action, Tup(Const(1)))
    #dels = dSda.variables
    dSda_vars, dSda = reverse_deriv(action, Tup(Const(1)))
    dSda = simplify(dSda)
    print(dSda_vars)


def solve_teg(prob: bc.BilliardsProblem, a:ITeg) -> Optional[bc.Path]:
    start = prob.tee
    walls = prob.walls
    end = prob.pocket
    p0 = np.array([start.x, start.y])
    p1 = np.array([end.x, end.y])
    mint = start.t
    maxt = end.t

    bounds = []
    ts = []
    xs = []
    ps = []

    for idx, w in enumerate(walls):
        x0, y0 = w.x0, w.y0
        x1, y1 = w.x1, w.y1
        wnorm = np.array([w.normalx, w.normaly])

        t = Var(f't{idx}', mint + (maxt - mint) * (idx + 1) / (len(walls) + 1))
        # x = Var(f'x{idx}', x1-0.01)
        x = Var(f'x{idx}', (x0 + x1) / 2)
        # x = Var(f'x{idx}', (x0 + x1) / 2)

        # m = (y1 - y0) / (x1 - x0)
        # b = y0 - x0 * (y1 - y0) / (x1 - x0)
        # y = m * x + b
        y = remap(x, x0, x1, y0, y1)

        p = np.array([x, y])

        bounds.append([x0, x1])
        ts.append(t)
        xs.append(x)
        ps.append(p)

    params = ts + xs

    from scipy.optimize import LinearConstraint
    lin_constraints = []
    constraints = []
    boundts = []
    boundxs = []

    eps = 0.0

    if len(ts) > 0:
        for _ in range(len(ts)):
            boundts.append((mint + eps, maxt - eps))

    if len(ts) > 1:
        # eps <= [-1, 1, 0, 0, 0, 0] [t0,   <= np.inf
        # eps <= [0, -1, 1, 0, 0, 0]  t1,   <= np.inf
        #                             t2,
        #                             x0, x1, x2]  (assumes params pack ts first)
        lincon_lbs = np.full(len(ts) - 1, eps)
        lincon_A = np.eye(len(ts) - 1, len(params), 1) - np.eye(len(ts) - 1, len(params))
        lincon_ubs = np.full_like(lincon_lbs, np.inf)
        lin_constraints.append(LinearConstraint(lincon_A, lincon_lbs, lincon_ubs))

    for x, xbs in zip(xs, bounds):
        x0, x1 = xbs
        boundxs.append((x0, x1))

    boundps = boundts + boundxs

    # params = [v for tx in zip(ts, xs) for v in tx]
    t = TegVar('t')
    m = 1  #Var('m', 1)
    x = linspline(t, [mint, *ts, maxt], [p0[0], *[p[0] for p in ps], p1[0]])
    y = linspline(t, [mint, *ts, maxt], [p0[1], *[p[1] for p in ps], p1[1]])

    potential = 0

    # vx = FwdDeriv(x, [(t, 1)]).deriv_expr
    # vy = FwdDeriv(y, [(t, 1)]).deriv_expr
    vx = simplify(reduce_to_base(fwd_deriv(x, [(t, 1)])))
    vy = simplify(reduce_to_base(fwd_deriv(y, [(t, 1)])))
    # v = FwdDeriv(x, [(t, 1), *[(_, 0) for _ in params]]).deriv_expr
    # kinetic = 1/2*m*(vx*vx + vy*vy)
    # lagrangian = kinetic - potential
    # action = Teg(mint, maxt, lagrangian, t)
    action = linspline_action(t, [mint, *ts, maxt],
                              [p0[0], *[p[0] for p in ps], p1[0]],
                              [p0[1], *[p[1] for p in ps], p1[1]], m)
    timing_prev = time.time()
    print(f'started constructing: dSda')
    # dSda = RevDeriv(action, Tup(Const(1)), output_list=params)
    # dels = dSda.variables
    dSda_vars, dSda_wdelta = reverse_deriv(action, Tup(Const(1)), output_list=params)
    sdSda_wdelta = simplify(dSda_wdelta)
    saction = simplify(action)
    sdSda = simplify(reduce_to_base(sdSda_wdelta))
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

    sddpSdas_sparse = [simplify(reverse_deriv(sdpSda, Tup(Const(1)), output_list=nonzero_hessian_params(param_num))[1]) for param_num, sdpSda in enumerate(sdpSdas)]

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
        print(f'    p: {k}')
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
            expand_sparse_hessian_row(param_num_, evaluate(sddpSda_sparse, {p_: v for p_, v in zip(params, vs)},
                                                           num_samples=args.num_samples, backend=args.backend))
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
    # print(f'init ddaction: {dd_action_func(init_guess)}')
    # print(d_action_func(init_guess))
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
    eps = 0.1
    # res = spop.minimize(action_func, init_guess, jac=d_action_func, method='BFGS')
    # res = spop.minimize(action_func, init_guess, jac=d_action_func, constraints=lin_constraints, bounds=boundps)
    res = spop.minimize(action_func, init_guess, jac=d_action_func, hess=dd_action_func, method='trust-constr', constraints=lin_constraints, bounds=boundps, options={'verbose': 1})
    # res = spop.minimize(action_func, init_guess, jac=d_action_func, method='L-BFGS-B')
    # res = spop.minimize(action_func, init_guess, jac=d_action_func, method='SLSQP')
    # res = spop.minimize(action_func, init_guess, jac=d_action_func, method='TNC', bounds=[b for bs in [((mint+eps, maxt-eps), (walls[i].x0, walls[i].x1)) for i in range(len(xs))] for b in bs])
    # res = spop.minimize(action_func, init_guess, jac=d_action_func, constraints=cons)  # TODO bounds?? threshold??

    print(f'res: {res.x}')
    print(f'final   action: {action_func(res.x)}')
    print(f'final  daction: {d_action_func(res.x)}')
    # print(f'final ddaction: {dd_action_func(res.x)}')
    print(f'  took: {time.time() - timing_prev}')

    tvals = []
    pvals = []
    for tval, xval, xvar, p in zip(res.x[:len(ts)], res.x[len(ts):], xs, ps):
        tvals.append(tval)
        yval = evaluate(p[1], {xvar: xval}, num_samples=args.num_samples, backend=args.backend)
        pvals.append(np.array([xval, yval]))
    path = bc.LinearPath([mint, *tvals, maxt], [p0, *pvals, p1])
    bind_param(x, res.x)
    bind_param(y, res.x)
    bind_param(vx, res.x)
    bind_param(vy, res.x)
    bind_param(action, res.x)
    bind_param(dSda, res.x)
    return path, substitute.substitute(x, t, a), substitute.substitute(y, t, a), substitute.substitute(vx, t, a), substitute.substitute(vy, t, a), action, dSda, params[0], params[1]


class Args(Tap):
    pixel_width: int = 30
    pixel_height: int = 30
    num_samples: int = 100
    t_samples: int = 400
    # backend: str = 'numpy'
    backend: str = 'C'  # TODO backend are different???


if __name__ == "__main__":
    start = bc.Tee(0, 0, 0)
    w0 = bc.Wall(-4, 0, 8, 12, 1, -1)
    w1 = bc.Wall(-5, 0, 14, 0, 0, 1)
    w2 = bc.Wall(14, 0, 16, 12, -6, 1)  # cant handle vertical walls yet; TODO reparameterize x
    w3 = bc.Wall(8, 12, 16, 12, 0, -1)  # needs to have x.lb <= x.ub right now for bounds
    w4 = bc.Wall(7, 8, 12, 8, 0, 1)

    w5 = bc.Wall(-4, 0, 12, 16)
    w6 = bc.Wall(0, -4, 16, 12)

    end = bc.Pocket(10, 10, 10)
    walls = [
        # w0,
        # w1,
        # w2,
        # w3,
        # w4,

        w5,
        w6,
        w5,
        w6,
        w5,
        w6,
        w5,
        w6,
        w5,
        w6,

        w5,
        w6,
        w5,
        w6,
        w5,
        w6,
        w5,
        w6,
        w5,
        w6,
    ]
    prob = bc.BilliardsProblem(start, walls, end)
    a = Var('a', 0)
    ans, x, y, vx, vy, action, dSda, t0, x0 = solve_teg(prob, a)
    print(f'ans: {ans}')

    args = Args()
    fig, axes = plt.subplots(nrows=2, ncols=2)


    def sample_expr(expr, ts_, v=a):
        for t_ in ts_:
            print(t_)
            expr.bind_variable(v, t_)
            f = evaluate(expr, num_samples=args.num_samples, backend=args.backend)
            # print(t_, f)
            yield f

    ts = np.array([start.t + (end.t - start.t) * i/args.t_samples for i in range(args.t_samples + 1)])

    xs = list(sample_expr(x, ts))
    ys = list(sample_expr(y, ts))
    vxs = list(sample_expr(vx, ts))
    vys = list(sample_expr(vy, ts))

    # bill_plot = axes[0]
    bill_plot = axes[0][0]
    bill_plot.plot(xs, ys)
    for w in walls:
        bill_plot.plot(np.array([w.x0, w.x1]), np.array([w.y0, w.y1]))
    bill_plot.set_aspect('equal', adjustable='box')

    axes[1][0].plot(ts, xs)
    axes[1][0].plot(ts, ys)
    axes[1][0].plot(ts, [math.sqrt(vx * vx + vy * vy) for vx, vy in zip(vxs, vys)])

    #
    # axes[1][1].plot(ts, vxs)
    # axes[1][1].plot(ts, vys)

    x0s = np.array([10 * i/args.t_samples for i in range(args.t_samples + 1)])

    from mpl_toolkits.mplot3d import axes3d
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

    # plt.xlim(0, 12)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


