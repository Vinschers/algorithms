from typing import Callable, Optional
import numpy as np


def prox_step(
    x: np.ndarray,
    grad_x: np.ndarray,
    prox_g: Callable,
    gamma: float,
):
    z = x - gamma * grad_x
    x_new = prox_g(z, gamma)
    return x_new, gamma


def prox_line_search_step(
    x: np.ndarray,
    f: Callable,
    grad_x: np.ndarray,
    prox_g: Callable,
    gamma: float,
):
    f_x = f(x)

    while True:
        z = x - gamma * grad_x
        next_x = prox_g(z, gamma)

        diff = next_x - x
        sq_dist = np.sum(diff**2)

        rhs = f_x + np.dot(grad_x, diff) + sq_dist / (2 * gamma)

        if f(next_x) <= rhs or gamma < 1e-12:
            break

        gamma /= 2

    gamma *= 1.2
    return next_x, gamma


def proximal_gradient(
    f,
    grad_f,
    g,
    prox_g,
    x0: np.ndarray,
    step_size: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = False,
):
    x = x0.copy()
    loss = []

    if step_size is None:
        gamma = 1.0
    else:
        gamma = step_size

    for _ in range(max_iter):
        grad_x = grad_f(x)

        if step_size is None:
            x_new, gamma = prox_line_search_step(x, f, grad_x, prox_g, gamma)
        else:
            x_new, gamma = prox_step(x, grad_x, prox_g, gamma)

        F_new = f(x_new) + g(x_new)
        loss.append(F_new.item())

        F_old = f(x) + g(x)
        norm_F = np.abs(F_old - F_new)

        norm_grad = np.linalg.norm((x - x_new) / gamma)

        x = x_new

        if norm_F < tol or norm_grad < epsilon:
            break

    if return_loss:
        return x, loss

    return x
