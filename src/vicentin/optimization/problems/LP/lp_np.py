import numpy as np
from vicentin.optimization.minimization import barrier_method


def LP(
    c: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 10,
    return_dual: bool = False,
):
    n = c.size
    H = np.zeros((n, n))

    obj = lambda x: c.dot(x)
    grad_obj = lambda x: c
    hess_obj = lambda x: H

    if A is None or b is None:
        equality = None
    else:
        equality = (A, b)

    constraints = []

    if G is not None and h is not None:
        for i in range(len(h)):
            g_row = G[i]
            h_val = h[i]

            const = lambda x, g_row=g_row, h_val=h_val: g_row.dot(x) - h_val
            grad_const = lambda x, g_row=g_row: g_row
            hess_const = lambda x, H=H: H

            constraints.append((const, grad_const, hess_const))

    x_star, (lambdas, nu_vec) = barrier_method(
        (obj, grad_obj, hess_obj),
        constraints,
        x0,
        equality,
        max_iter,
        epsilon,
        mu,
        return_dual=True,
    )

    return (x_star, (lambdas, nu_vec)) if return_dual else x_star
