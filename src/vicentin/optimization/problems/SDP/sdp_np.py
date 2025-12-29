from typing import Any, Callable, Sequence, Tuple

import numpy as np
from numpy.linalg import inv, solve, LinAlgError, slogdet
from vicentin.optimization.minimization import barrier_method


def sdp_linear_solver(
    hess_f: Callable,
    grad_f: Callable,
    x: np.ndarray,
    w: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n = int(np.sqrt(x.size))
    m = A.shape[0]

    X = x.reshape((n, n))

    grad = grad_f(x)

    g = grad + A.T @ w
    h = A @ x - b

    delta_w = np.zeros_like(w)

    inv_H_A = (X @ A.reshape((m, n, n)) @ X).transpose(1, 2, 0).reshape(-1, m)
    inv_H_g = np.ravel(X @ g.reshape((n, n)) @ X)

    S = -A @ inv_H_A

    delta_w = solve(S, A @ inv_H_g - h)

    rhs_x = -A.T @ delta_w - g
    rhs_X = rhs_x.reshape((n, n))

    delta_X = X @ rhs_X @ X
    delta_X = (delta_X + delta_X.T) / 2.0

    delta_x = delta_X.ravel()

    M = rhs_X @ X
    decrement_squared = np.sum(M * M.T)

    return delta_x, delta_w, decrement_squared


def SDP(
    C: np.ndarray,
    equality_constraints: Sequence[Any],
    X0: np.ndarray,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
):
    n = X0.shape[0]
    m = len(equality_constraints)

    C = (C + C.T) / 2.0

    A_list = []
    b_list = []
    for A_i, b_i in equality_constraints:
        A_list.append((A_i + A_i.T) / 2.0)
        b_list.append(b_i)

    f = lambda x: np.dot(C.ravel(), x)
    grad_f = lambda x: C.ravel()
    hess_f = lambda x: np.zeros((n**2, n**2))

    F = (f, grad_f, hess_f)

    if m > 0:
        A = np.array([a.flatten() for a in A_list])
        b = np.array(b_list)
        equality = (A, b)
    else:
        equality = None

    def psd_inequality(x):
        X = x.reshape((n, n))

        try:
            sign, logdet = slogdet(X)
            if sign <= 0 or np.any(np.linalg.eigvals(X) < 0):
                return np.inf

            return -logdet
        except LinAlgError:
            return np.inf

    def psd_grad(x):
        X = x.reshape((n, n))
        try:
            return -inv(X).flatten()
        except LinAlgError:
            return np.zeros(n**2)

    G = [(psd_inequality, psd_grad, None, 1)]

    x_star = barrier_method(
        F,
        G,
        X0.flatten(),
        equality,
        max_iter,
        epsilon,
        mu,
        linear_solver=sdp_linear_solver,
    )

    return x_star.reshape((n, n))
