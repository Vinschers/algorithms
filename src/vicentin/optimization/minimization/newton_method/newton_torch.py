from typing import Callable, Optional

import torch
from torch.func import jacrev, hessian
from torch.linalg import norm, ldl_factor, ldl_solve


def backtrack_line_search(
    f: Callable,
    r: Callable,
    x: torch.Tensor,
    delta_x: torch.Tensor,
    w: torch.Tensor,
    delta_w: torch.Tensor,
    alpha: float,
    beta: float,
):
    y = r(x, w).item()
    t = 1

    while True:
        f_x = f(x + t * delta_x)
        infeasible = torch.isnan(f_x) or torch.isinf(f_x)
        feasible = not infeasible

        if feasible:
            lipschitz = (
                r(x + t * delta_x, w + t * delta_w).item()
                <= (1 - alpha * t) * y
            )

            if lipschitz:
                break

        t *= beta

        if t < 1e-12:
            break

    return t


def solve_kkt_system(kkt_matrix, kkt_rhs, n, x_shape):
    kkt_rhs = kkt_rhs.unsqueeze(-1)

    LD, pivots = ldl_factor(kkt_matrix)
    delta_x_w = ldl_solve(LD, pivots, kkt_rhs).flatten()
    delta_x, delta_w = delta_x_w[:n].view(x_shape), delta_x_w[n:]

    return delta_x, delta_w


def newton_step(
    f: Callable,
    grad_f: Callable,
    hess_f: Callable,
    r: Callable,
    x: torch.Tensor,
    w: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 0.25,
    beta: float = 0.5,
):
    m = A.shape[0]
    n = x.numel()

    gradient = grad_f(x).view(-1)
    H = hess_f(x).view(n, n)

    kkt_matrix = torch.cat(
        (
            torch.cat((H, A.T), dim=1),
            torch.cat((A, torch.zeros((m, m))), dim=1),
        ),
        dim=0,
    )
    kkt_rhs = -torch.cat((gradient + A.T @ w, A @ x.view(-1) - b))

    delta_x = torch.zeros_like(x)
    delta_w = torch.zeros_like(w)

    try:
        delta_x, delta_w = solve_kkt_system(kkt_matrix, kkt_rhs, n, x.shape)
    except RuntimeError:
        H = H + 1e-9 * torch.eye(n, device=H.device, dtype=H.dtype)
        kkt_matrix = torch.cat(
            (
                torch.cat((H, A.T), dim=1),
                torch.cat((A, torch.zeros((m, m))), dim=1),
            ),
            dim=0,
        )

        try:
            delta_x, delta_w = solve_kkt_system(kkt_matrix, kkt_rhs, n, x.shape)
        except RuntimeError:
            print("Could not solve for \\Delta_x.")

    t = backtrack_line_search(f, r, x, delta_x, w, delta_w, alpha, beta)

    x = x + t * delta_x
    w = w + t * delta_w

    decrement_squared = delta_x.view(-1) @ H @ delta_x.view(-1)

    return x, w, decrement_squared


def newton(
    f: Callable,
    x0: torch.Tensor,
    equality: Optional[tuple] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    epsilon: float = 1e-4,
    alpha: float = 0.25,
    beta: float = 0.5,
    return_dual: bool = False,
    return_loss: bool = False,
):
    x = x0.clone().detach().to(torch.float64)
    i = 0
    loss = []

    grad_f_raw = jacrev(f)
    hess_f_raw = hessian(f)

    def grad_f(x):
        g = grad_f_raw(x)
        if isinstance(g, (list, tuple)):
            return g[0]
        return g

    def hess_f(x):
        H = hess_f_raw(x)
        if isinstance(H, (list, tuple)):
            return H[0]
        return H

    device = x.device
    dtype = x.dtype

    if equality is None:
        A = torch.empty((0, x.numel()), device=device, dtype=dtype)
        b = torch.empty(0, device=device, dtype=dtype)
        w = torch.empty(0, device=device, dtype=dtype)
    else:
        A, b = equality
        w = torch.zeros(A.shape[0], device=device, dtype=dtype)

    r = lambda x, w: norm(
        torch.cat([grad_f(x).view(-1) + A.T @ w, A @ x.view(-1) - b])
    )

    with torch.no_grad():
        y_new = f(x).item()

    while True:
        y = y_new
        if y == torch.inf:
            raise ValueError(f"Reached infeasible point: {x}.")

        with torch.no_grad():
            x, w, decrement_squared = newton_step(
                f, grad_f, hess_f, r, x, w, A, b, alpha, beta
            )

            y_new = f(x).item()
            r_val = r(x, w).item()

        loss.append(y_new)

        feasible = torch.isclose(A @ x.view(-1), b).all()

        if (not feasible and r_val <= epsilon) or (
            feasible and decrement_squared <= 2 * epsilon
        ):
            break

        if abs(y_new - y) < tol:
            break

        i += 1

        if i >= max_iter:
            break

    output = [x]

    if return_dual:
        output.append(w)

    if return_loss:
        output.append(loss)

    if len(output) == 1:
        return output[0]

    return output
