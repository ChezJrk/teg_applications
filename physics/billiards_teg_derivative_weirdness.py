from typing import List, Tuple, Optional
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
from teg.eval import evaluate
from teg.passes import simplify
from teg.passes import substitute
from tap import Tap
from teg.passes.simplify import simplify


def remap(t, t0, t1, x0, x1):
    s = (t - t0) * (1 / (t1 - t0))
    return x0 + s * (x1 - x0)


def linspline(t: ITeg, ts: List[ITeg], xs: List[ITeg]):
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

    v = FwdDeriv(x, [(t_, 1), *[(_, 0) for _ in params]]).deriv_expr
    kinetic = 1 / 2 * m * v * v
    lagrangian = kinetic - potential
    action = Teg(0, 2, lagrangian, t_)
    dSda = RevDeriv(action, Tup(Const(1)))
    dels = dSda.variables
    print(dels)

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

    v = FwdDeriv(x, [(t, 1), *[(_, 0) for _ in params]]).deriv_expr
    kinetic = 1 / 2 * m * v * v
    lagrangian = kinetic - potential
    action = Teg(0, 1, lagrangian, t)
    dSda = RevDeriv(action, Tup(Const(1)))
    dels = dSda.variables


def solve_teg(prob: bc.BilliardsProblem, a: ITeg) -> Optional[bc.Path]:
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
        x = Var(f'x{idx}', (x0 + x1) / 2 + 1)
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

    constraints = []

    eps = 0.01

    if len(ts) > 0:
        constraints.append(ts[0] - eps)
        constraints.append(1 - eps - ts[-1])

    for t0, t1 in zip(ts, ts[1:]):
        constraints.append(t1 - t0 - eps)

    for x, xbs in zip(xs, bounds):
        x0, x1 = xbs
        constraints.append(x1 - x)
        constraints.append(x - x0)

    params = ts + xs
    t = TegVar('t')
    m = 1  # Var('m', 1)
    x = linspline(t, [mint, *ts, maxt], [p0[0], *[p[0] for p in ps], p1[0]])
    y = linspline(t, [mint, *ts, maxt], [p0[1], *[p[1] for p in ps], p1[1]])

    potential = 0

    vx = FwdDeriv(x, [(t, 1)]).deriv_expr
    vy = FwdDeriv(y, [(t, 1)]).deriv_expr
    # v = FwdDeriv(x, [(t, 1), *[(_, 0) for _ in params]]).deriv_expr
    kinetic = 1 / 2 * m * (vx * vx + vy * vy)
    lagrangian = kinetic - potential
    scale_factor = 1
    action = Teg(mint, maxt, lagrangian, t) * scale_factor
    dSda = RevDeriv(action, Tup(Const(1)))
    dels = dSda.variables
    print(f'dSda vars: {dels}')  # TODO why is ddt a var? twice?
    saction = simplify(action)
    sdSda = simplify(dSda.deriv_expr)

    args = Args()

    def bind_param(expr, vs):
        for param, v in zip(params, vs):
            expr.bind_variable(param, v)

    def action_func(vs):
        bind_param(action, vs)
        k = evaluate(saction, num_samples=args.num_samples, backend=args.backend)
        print(f'  param: {vs}\t{k}')
        return k

    def d_action_func(vs):
        bind_param(dSda, vs)
        k = evaluate(sdSda, num_samples=args.num_samples, backend=args.backend)
        print(f'  dparm: {vs}\t{k[0:len(params)]}')
        return k[:len(params)]

    init_guess = [p.value for p in params]
    print(f'init guess: {init_guess}')
    print(f'init action: {action_func(init_guess)}')
    print(f'init daction: {d_action_func(init_guess)}')
    # print(d_action_func(init_guess))
    cons = []
    for cteg in constraints:
        def cteg_func(param_vals):
            bind_param(cteg, param_vals)
            k = evaluate(cteg, num_samples=args.num_samples, backend=args.backend)
            return k

        cons.append({
            'type': 'ineq',
            'fun': cteg_func
        })
    print(f'cons: {list(map(str, constraints))}')
    print(f'cons2: {cons}')
    res = spop.minimize(action_func, init_guess, jac=d_action_func)
    # res = spop.minimize(action_func, init_guess, method='trust-constr', constraints=cons)

    print(f'res: {res.x}')
    print(f'final action: {action_func(res.x)}')
    print(f'final daction: {d_action_func(res.x)}')

    tvals = []
    xvals = []
    for tval, xval in zip(res.x[::2], res.x[1::2]):
        tvals.append(tval)
        xvals.append(xval)
    path = bc.LinearPath([mint, *tvals, maxt], [p0, *ps, p1])
    bind_param(x, res.x)
    bind_param(y, res.x)
    bind_param(vx, res.x)
    bind_param(vy, res.x)
    bind_param(action, res.x)
    bind_param(dSda, res.x)
    print(f'backend diff | saction: {np.array(evaluate(saction, backend="C")) - evaluate(saction, backend="numpy")}')
    print(f'backend diff | sdSda:   {np.array(evaluate(sdSda, backend="C")) - evaluate(sdSda, backend="numpy")}')
    print(f'simplify diff | action: {np.array(evaluate(saction, backend="C")) - evaluate(action, backend="C")}')
    print(f'simplify diff | dSda:   {np.array(evaluate(sdSda, backend="C")) - evaluate(dSda, backend="C")}')
    return path, substitute.substitute(x, t, a), substitute.substitute(y, t, a), substitute.substitute(vx, t,
                                                                                                       a), substitute.substitute(
        vy, t, a), action, dSda, params[0], params[1]


class Args(Tap):
    pixel_width: int = 30
    pixel_height: int = 30
    num_samples: int = 100
    t_samples: int = 100
    # backend: str = 'numpy'
    backend: str = 'C'  # TODO backend are different???


if __name__ == "__main__":
    start = bc.Tee(0, 0, 0)
    w0 = bc.Wall(-2, 2, 8, 12, 1, -1)
    w1 = bc.Wall(1, 0, 14, 0, 0, 1)
    end = bc.Pocket(10, 10, 10)
    # w0 = bc.Wall(0, 10, 10, 10, 0, -1)
    # end = bc.Pocket(10, 1)
    walls = [
        w0,
        # w1,
    ]
    prob = bc.BilliardsProblem(start, walls, end)
    a = Var('a', 0)
    ans, x, y, vx, vy, action, dSda, t0, x0 = solve_teg(prob, a)
    print(ans)

    args = Args()
    fig, axes = plt.subplots(nrows=2, ncols=2)


    def sample_expr(expr, ts_, v=a):
        for t_ in ts_:
            print(t_)
            expr.bind_variable(v, t_)
            f = evaluate(expr, num_samples=args.num_samples, backend=args.backend)
            # print(t_, f)
            yield f


    ts = np.array([start.t + (end.t - start.t) * i / args.t_samples for i in range(args.t_samples + 1)])

    xs = list(sample_expr(x, ts))
    ys = list(sample_expr(y, ts))
    vxs = list(sample_expr(vx, ts))
    vys = list(sample_expr(vy, ts))

    # bill_plot = axes[0]
    bill_plot = axes[0][0]
    bill_plot.plot(xs, ys)
    for w in walls:
        bill_plot.plot(np.array([w.x0, w.x1]), np.array([w.y0, w.y1]))

    axes[1][0].plot(ts, xs)
    axes[1][0].plot(ts, ys)
    axes[1][0].plot(ts, [math.sqrt(vx * vx + vy * vy) for vx, vy in zip(vxs, vys)])

    #
    # axes[1][1].plot(ts, vxs)
    # axes[1][1].plot(ts, vys)

    x0s = np.array([10 * i / args.t_samples for i in range(args.t_samples + 1)])

    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    # X, Y, Z = axes3d.get_test_data(0.05)

    buff = 10

    # vals = []
    # atemp = simplify(action)
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

    vals = []
    atemp = simplify(dSda.deriv_expr)
    for t_ in ts[::]:
        action.bind_variable(t0, t_)
        actions = np.array(list(sample_expr(atemp, x0s, v=x0)))
        vals.append(actions)
        print(f'\t{t_}')
    vals = np.array(vals)
    ax.plot_surface(np.array([x0s for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :, 0], rstride=2, cstride=1, alpha=0.5)
    ax.contour(np.array([x0s for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :, 0], levels=[_ for _ in np.arange(-10, 6, 1)], cmap=cm.get_cmap('magma'), linestyles="solid")
    ax.set_xlabel('x0')
    ax.set_ylabel('t0')
    ax.set_zlabel('dt0')
    ax2.plot_surface(np.array([x0s[:] for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s[:]]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :, 1], rstride=2, cstride=1, alpha=0.5)
    ax2.contour(np.array([x0s[:] for _ in ts[buff:-buff]]), np.array([np.array([t_ for _ in x0s[:]]) for t_ in ts[buff:-buff]]), vals[buff:-buff, :, 1], levels=[_ for _ in np.arange(-3, 5, 0.5)], cmap=cm.get_cmap('magma'), linestyles="solid")
    ax2.set_xlabel('x0')
    ax2.set_ylabel('t0')
    ax2.set_zlabel('dx0')

    actions = np.array(list(sample_expr(substitute.substitute(action, t0, a), ts)))
    dSdas = np.array(list(sample_expr(substitute.substitute(action, x0, a), xs)))
    # dSdas = np.array(list(sample_expr(substitute.substitute(dSda.deriv_expr, t0, a), ts)))
    axes[0][1].plot(ts, actions)
    axes[1][1].plot(ts, dSdas)

    # plt.xlim(0, 12)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


