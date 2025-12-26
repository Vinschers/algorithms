from typing import Callable

import torch
from torch.func import jacrev, hessian


def backtrack_line_search(
    f: Callable,
    y: float,
    grad_x: torch.Tensor,
    x: torch.Tensor,
    delta_x: torch.Tensor,
    alpha: float,
    beta: float,
):
    t = 1
    slope = alpha * torch.vdot(grad_x, delta_x)

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
    x0: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-5,
    epsilon: float = 1e-4,
    alpha: float = 0.25,
    beta: float = 0.5,
    return_loss: bool = False,
):
    x = x0.clone().detach().to(torch.float32)
    decrement_squared = torch.inf
    i = 0
    loss = []

    grad_f = jacrev(f)
    hess_f = hessian(f)

    y_new = f(x)

    while decrement_squared > 2 * epsilon:
        y = y_new
        gradient = grad_f(x)
        H = hess_f(x)

        if isinstance(gradient, (list, tuple)):
            gradient = gradient[0]

        if y == torch.inf:
            raise ValueError(f"Reached infeasible point: {x}.")

        try:
            delta_nt = torch.linalg.solve(H, -gradient)
        except RuntimeError:
            print("Hessian is singular or not positive definite.")
            break

        decrement_squared = torch.vdot(delta_nt, -gradient)

        with torch.no_grad():
            t = backtrack_line_search(f, y, gradient, x, delta_nt, alpha, beta)

        x = x + t * delta_nt
        y_new = f(x)

        loss.append(y_new.cpu().item())

        if torch.abs(y_new - y) < tol:
            break

        i += 1

        if i >= max_iter:
            break

    if return_loss:
        return x, loss

    return x
