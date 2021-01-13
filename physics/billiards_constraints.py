from typing import Union, Optional
import collections
import numpy as np
import pprint
import textwrap


# inputs

class Wall(object):
    def __init__(self, x0, y0, x1, y1, normalx=None, normaly=None):
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.normalx = normalx if normalx is not None else -(y1 - y0)
        self.normaly = normaly if normalx is not None else x1 - x0

    def __repr__(self):
        return f'{self.__class__.__name__}(x0={self.x0!r}, y0={self.y0!r}, x1={self.x1!r}, y1={self.y1!r})'


class Tee(object):
    def __init__(self, x, y, t=0):
        super().__init__()
        self.x = x
        self.y = y
        self.t = t

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x!r}, y={self.y!r}, t={self.t!r})'


class Pocket(object):
    def __init__(self, x, y, t=1):
        super().__init__()
        self.x = x
        self.y = y
        self.t = t

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x!r}, y={self.y!r}, t={self.t!r})'


class BilliardsProblem(object):
    def __init__(self, tee: Tee, walls: [Wall], pocket: Pocket):
        super().__init__()
        self.tee = tee
        self.walls = walls
        self.pocket = pocket

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n'
            f'  tee={self.tee!r},\n'
            f'  walls={self.walls!r},\n'
            f'  pocket={self.pocket!r},\n'
            f')'
        )


# outputs

class Path(object):
    def __init__(self):
        super().__init__()

    def interpolate(self, t):
        raise NotImplementedError


class LinearPath(Path):
    def __init__(self, ts, xs):
        super().__init__()
        assert len(ts) > 0
        assert len(ts) == len(xs)
        self.ts = ts
        self.xs = xs

    def interpolate(self, t):
        if t < self.ts[0]:
            return self.xs[0]
        for t0, t1, x0, x1 in zip(self.ts, self.ts[1:], self.xs, self.xs[1:]):
            if t <= t1:
                return remap(t, t0, t1, x0, x1)
        return self.xs[-1]

    def __str__(self):
        return f'LinearPath: ts={self.ts}, xs={self.xs}'


# helpers

def remap(t, t0, t1, x0, x1):
    s = (t - t0) / (t1 - t0)
    return x0 + s*(x1 - x0)


# solvers

def solve_analytic(prob: BilliardsProblem) -> Optional[Path]:
    start = prob.tee
    w0 = prob.walls[0]
    end = prob.pocket

    def flip_over_wall(p, w: Wall):
        a = np.array([w.x0, w.y0])
        b = np.array([w.x1, w.y1])
        t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
        proj_p_on_wall = a + t * (b - a)
        return 2 * proj_p_on_wall - p

    def on_correct_side(p, w: Wall):
        n = np.array([w.normalx, w.normaly])
        a = np.array([w.x0, w.y0])
        return np.dot(p - a, n) > 0

    p0 = np.array([start.x, start.y])
    p1 = np.array([end.x, end.y])

    if not on_correct_side(p0, w0) or not on_correct_side(p1, w0):
        return None

    a = np.array([w0.x0, w0.y0])
    b = np.array([w0.x1, w0.y1])

    def intersect_line_segments(a0, b0, a1, b1):
        d0 = b0 - a0
        d1 = b1 - a1
        denom = np.cross(d0, d1)
        if denom == 0:
            return None
        factor = (a1 - a0) / denom
        t = np.cross(factor, d0)
        s = np.cross(factor, d1)
        if 0 <= t <= 1 and 0 <= s <= 1:
            return a0 + t * d0, t
        return None

    isect_result = intersect_line_segments(a, b, p0, flip_over_wall(p1, w0))
    if not isect_result:
        return None
    isect, isect_t = isect_result

    return LinearPath(ts=[0, isect_t, 1], xs=[p0, isect, p1])


def main():
    start = Tee(0, 0)
    w0 = Wall(-1, 1, 9, 11, 1, -1)
    walls = [w0]
    end = Pocket(10, 10)
    prob = BilliardsProblem(start, walls, end)
    ans = solve_analytic(prob)
    print(ans)


if __name__ == '__main__':
    main()












