from typing import Callable, Optional

import numpy as np
from numpy.linalg import norm, solve


def backtrack_line_search(
    f: Callable,
    r: Callable,
    x: np.ndarray,
    delta_x: np.ndarray,
    w: np.ndarray,
    delta_w: np.ndarray,
    alpha: float,
    beta: float,
):
    y = r(x, w)
    t = 1

    while True:
        f_x = f(x + t * delta_x)
        infeasible = np.isnan(f_x) or np.isinf(f_x)
        feasible = not infeasible

        if feasible:
            r_val = r(x + t * delta_x, w + t * delta_w)

            lipschitz = r_val <= (1 - alpha * t) * y

            if lipschitz:
                break

        t *= beta

        if t < 1e-12:
            break

    return t


def newton_step(
    f: Callable,
    grad_f: Callable,
    hess_f: Callable,
    r: Callable,
    x: np.ndarray,
    w: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.25,
    beta: float = 0.5,
):
    n = x.size
    m = A.shape[0]

    gradient = grad_f(x).ravel()
    H = hess_f(x).reshape(n, n)

    kkt_matrix = np.block([[H, A.T], [A, np.zeros((m, m))]])
    kkt_rhs = -np.block([gradient + A.T @ w, A @ x.ravel() - b])

    delta_x = np.zeros_like(x)
    delta_w = np.zeros_like(w)

    try:
        delta_x_w = solve(kkt_matrix, kkt_rhs)
        delta_x, delta_w = delta_x_w[:n].reshape(x.shape), delta_x_w[n:]
    except RuntimeError:
        print("Could not solve for \\Delta_x.")

    t = backtrack_line_search(f, r, x, delta_x, w, delta_w, alpha, beta)

    x = x + t * delta_x
    w = w + t * delta_w

    decrement_squared = delta_x.ravel() @ H @ delta_x.ravel()

    return x, w, decrement_squared


def newton(
    f: Callable,
    grad_f: Callable,
    hess_f: Callable,
    x0: np.ndarray,
    equality: Optional[tuple] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    epsilon: float = 1e-4,
    alpha: float = 0.25,
    beta: float = 0.5,
    return_dual: bool = False,
    return_loss: bool = False,
):
    x = x0.copy()
    i = 0
    loss = []

    if equality is None:
        A = np.empty((0, x.size))
        b = np.empty(0)
        w = np.empty(0)
    else:
        A, b = equality
        w = np.zeros(A.shape[0])

    r = lambda x, w: norm(
        np.concatenate([grad_f(x).ravel() + A.T @ w, A @ x.ravel() - b])
    )

    y_new = f(x)

    while True:
        y = y_new
        if y == np.inf:
            raise ValueError(f"Reached infeasible point: {x}.")

        x, w, decrement_squared = newton_step(
            f, grad_f, hess_f, r, x, w, A, b, alpha, beta
        )

        y_new = f(x)
        r_val = r(x, w)

        loss.append(y_new)

        feasible = np.isclose(A @ x.ravel(), b).all()

        if (not feasible and r_val <= epsilon) or (
            feasible and decrement_squared <= 2 * epsilon
        ):
            break

        if np.abs(y_new - y) < tol:
            break

        i += 1

        if i >= max_iter:
            break

    output = []
    output.append(x)

    if return_dual:
        output.append(w)

    if return_loss:
        output.append(loss)

    if len(output) == 1:
        return output[0]

    return output
