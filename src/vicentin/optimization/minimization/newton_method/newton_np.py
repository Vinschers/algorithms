from typing import Callable
import numpy as np


def backtrack_line_search(
    f: Callable,
    y: float,
    grad_x: np.ndarray,
    x: np.ndarray,
    delta_x: np.ndarray,
    alpha: float,
    beta: float,
):
    t = 1
    slope = alpha * np.dot(grad_x, delta_x)

    while True:
        try:
            armijo = f(x + t * delta_x) <= y + t * slope

            if armijo or t < 1e-12:
                break

        except (ValueError, OverflowError):
            pass

        t *= beta

    return t


def newton(
    f: Callable,
    grad_f: Callable,
    hess_f: Callable,
    x0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-5,
    epsilon: float = 1e-4,
    alpha: float = 0.25,
    beta: float = 0.5,
    return_loss: bool = False,
):
    x = x0.copy()
    decrement_squared = np.inf
    i = 0
    loss = []

    y_new = f(x)

    while decrement_squared > 2 * epsilon:
        y = y_new
        gradient = grad_f(x)
        hessian = hess_f(x)

        if y == np.inf:
            raise ValueError(f"Reached infeasible point: {x}.")

        delta_nt = np.linalg.solve(hessian, -gradient)
        decrement_squared = np.dot(delta_nt, -gradient)

        t = backtrack_line_search(f, y, gradient, x, delta_nt, alpha, beta)

        x = x + t * delta_nt
        y_new = f(x)

        loss.append(y_new)

        if np.abs(y_new - y) < tol:
            break

        i += 1

        if i >= max_iter:
            break

    if return_loss:
        return x, loss

    return x
