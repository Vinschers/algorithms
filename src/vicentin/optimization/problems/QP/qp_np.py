import numpy as np
from vicentin.optimization.minimization import barrier_method


def QP(
    P: np.ndarray,
    q: np.ndarray,
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
    n = q.size

    obj = lambda x: 0.5 * x.dot(P @ x) + q.dot(x)
    grad_obj = lambda x: P @ x + q
    hess_obj = lambda x: P

    if A is None or b is None:
        equality = None
    else:
        equality = (A, b)

    constraints = []

    if G is not None and h is not None:
        H = np.zeros((n, n))

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
