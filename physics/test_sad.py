import matplotlib.pyplot as plt
from tqdm import trange
from scipy.optimize import minimize
import numpy as np
from typing import List

from teg import (
    ITeg,
    Const,
    Var,
    IfElse,
    Teg,
    Tup,
    LetIn,
    TegVar,
)
from teg.derivs.reverse_deriv import reverse_deriv
from teg.derivs import FwdDeriv, RevDeriv
from physics.smooth import InvertSqrt, IsNotNan
from teg.math.smooth import Invert, Sqrt
from teg.eval import evaluate
from teg.passes.substitute import substitute
from teg.passes.simplify import simplify
from teg.passes.reduce import reduce_to_base


class Args:
    n: int = 100
    backend: str = 'C'


def main():
    args = Args()

    t = Var('t')
    theta = TegVar()

    flo = 0
    fhi = 1
    f = IfElse(theta < t, flo, fhi)
    int_all = Teg(0, 1, f, theta)

    int_l = Teg(0, t, flo, theta)
    int_r = Teg(t, 1, fhi, theta)
    # int_split = int_l + int_r

    squint_all = int_all * int_all
    squint_split = int_l * int_l + int_r * int_r

    params = [t]
    param_vals = {t: 0.5}
    bad_args = {'ignore_deltas': True, 'ignore_bounds': True}
    _, squint_all_deriv = reverse_deriv(squint_all, Tup(Const(1)), output_list=params)
    _, squint_split_deriv = reverse_deriv(squint_split, Tup(Const(1)), output_list=params)
    _, squint_all_deriv_bad = reverse_deriv(squint_all, Tup(Const(1)), output_list=params, args=bad_args)
    _, squint_split_deriv_bad = reverse_deriv(squint_split, Tup(Const(1)), output_list=params, args=bad_args)

    squint_all_deriv = reduce_to_base(squint_all_deriv)
    squint_split_deriv = reduce_to_base(squint_split_deriv)

    squint_all_val = evaluate(squint_all_deriv, param_vals, num_samples=args.n, backend=args.backend)
    squint_split_val = evaluate(squint_split_deriv, param_vals, num_samples=args.n, backend=args.backend)
    squint_all_val_bad = evaluate(squint_all_deriv_bad, param_vals, num_samples=args.n, backend=args.backend)
    squint_split_val_bad = evaluate(squint_split_deriv_bad, param_vals, num_samples=args.n, backend=args.backend)

    print(f'squint_all_val:       {squint_all_val}')
    print(f'squint_split_val:     {squint_split_val}')
    print(f'squint_all_val_bad:   {squint_all_val_bad}')
    print(f'squint_split_val_bad: {squint_split_val_bad}')


if __name__ == "__main__":
    main()
