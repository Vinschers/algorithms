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
            return 0

    return t


def default_solver(
    hess_f: Callable,
    grad_f: Callable,
    x: np.ndarray,
    w: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
):
    grad = grad_f(x)
    H = hess_f(x)

    if np.linalg.svd(H, compute_uv=False).min() < 1e-12:
        H = H + 1e-8 * np.eye(H.shape[-1])

    g = grad + A.T @ w
    h = A @ x - b

    inv_H_A = solve(H, A.T)
    inv_H_g = solve(H, g)

    S = -A @ inv_H_A

    delta_w = solve(S, A @ inv_H_g - h)
    delta_x = solve(H, -A.T @ delta_w - g)

    decrement_squared = delta_x @ H @ delta_x

    return delta_x, delta_w, decrement_squared


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
    linear_solver: Optional[Callable] = None,
):
    if linear_solver is None:
        linear_solver = default_solver

    delta_x, delta_w, decrement_squared = linear_solver(
        hess_f, grad_f, x, w, A, b
    )

    t = backtrack_line_search(f, r, x, delta_x, w, delta_w, alpha, beta)

    x = x + t * delta_x
    w = w + t * delta_w

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
    linear_solver: Optional[Callable] = None,
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

    r = lambda x, w: norm(np.concatenate([grad_f(x) + A.T @ w, A @ x - b]))

    y_new = f(x)

    while True:
        y = y_new
        if y == np.inf:
            raise ValueError(f"Reached infeasible point: {x}.")

        x, w, decrement_squared = newton_step(
            f, grad_f, hess_f, r, x, w, A, b, alpha, beta, linear_solver
        )

        y_new = f(x)
        r_val = r(x, w)

        loss.append(y_new)

        feasible = np.isclose(A @ x, b).all()

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
