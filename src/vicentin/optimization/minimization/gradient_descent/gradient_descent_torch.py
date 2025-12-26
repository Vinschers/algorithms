from typing import Callable, Optional

import torch
from torch.func import jacrev


def gamma_step(
    x: torch.Tensor, f: Callable, grad_x: torch.Tensor, gamma: float
):
    return x - gamma * grad_x, gamma


def line_search_step(
    x: torch.Tensor, f: Callable, grad_x: torch.Tensor, gamma: float
):
    next_x = x - gamma * grad_x
    f_x = f(x)

    while True:
        armijo = f(next_x) <= f_x + torch.dot(grad_x, next_x - x) + torch.sum(
            (next_x - x) ** 2
        ) / (2 * gamma)

        if armijo or gamma < 1e-12:
            break

        gamma /= 2
        next_x = x - gamma * grad_x

    gamma *= 1.2  # Heuristic to speed up
    return x - gamma * grad_x, gamma


def gradient_descent(
    f,
    x0: torch.Tensor,
    step_size: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = False,
):
    x = x0.clone().detach()
    loss = []

    if step_size is None:
        step_func = line_search_step
        gamma = 1
    else:
        step_func = gamma_step
        gamma = step_size

    grad_f = jacrev(f)

    for _ in range(max_iter):

        grad_x = grad_f(x)
        if isinstance(grad_x, (list, tuple)):
            grad_x = grad_x[0]

        with torch.no_grad():
            x_new, gamma = step_func(x, f, grad_x, gamma)

        f_new = f(x_new)

        loss.append(f_new.item())

        norm_f = torch.abs(f(x) - f_new)
        norm_grad = torch.linalg.norm(grad_x)

        x = x_new

        if norm_f < tol or norm_grad < epsilon:
            break

    if return_loss:
        return x, loss

    return x
