from typing import Sequence
import numpy as np
from vicentin.optimization.minimization import barrier_method


def SOCP(
    f: np.ndarray,
    socp_constraints: Sequence[Sequence[np.ndarray]],
    F: np.ndarray,
    g: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_dual: bool = False,
):
    n = f.size

    obj = lambda x: f.dot(x)
    grad_obj = lambda x: f
    hess_obj = lambda x: np.zeros((n, n))

    if F is None or g is None:
        equality = None
    else:
        equality = (F, g)

    G = []
    for A, b, c, d in socp_constraints:
        A_T_A = A.T @ A
        A_T_b = A.T @ b
        c_c_T = np.outer(c, c)

        H = 2 * (A_T_A - c_c_T)

        const = (
            lambda x, A=A, b=b, c=c, d=d: np.sum((A @ x + b) ** 2)
            - (c.dot(x) + d) ** 2
        )

        grad_const = lambda x, A_T_A=A_T_A, A_T_b=A_T_b, c=c, d=d: 2 * (
            A_T_A @ x + A_T_b - (c.dot(x) + d) * c
        )
        hess_const = lambda x, H=H: H

        G.append((const, grad_const, hess_const))

        upper_cone = lambda x, c=c, d=d: -c.dot(x) - d
        upper_cone_grad = lambda x, c=c: -c
        upper_cone_hess = lambda x: np.zeros((n, n))

        G.append((upper_cone, upper_cone_grad, upper_cone_hess))

    x_star, (lambdas, mu_vec) = barrier_method(
        (obj, grad_obj, hess_obj),
        G,
        x0,
        equality,
        max_iter,
        epsilon,
        mu,
        return_dual=True,
    )

    u = []
    v = []

    for i in range(len(socp_constraints)):
        A, b, c, d = socp_constraints[i]
        l1 = lambdas[2 * i]  # Multiplier for quadratic form
        l2 = lambdas[2 * i + 1]  # Multiplier for (c^\top x + d) >= 0

        u_i = -2 * l1 * (A @ x_star + b)
        v_i = 2 * l1 * (c.dot(x_star) + d) + l2

        u.append(u_i)
        v.append(v_i)

    return (x_star, (u, v, mu_vec)) if return_dual else x_star
