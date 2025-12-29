from typing import Any, Callable, Sequence, Tuple
import math

import torch
from torch.linalg import slogdet, solve
from vicentin.optimization.minimization import barrier_method


def sdp_linear_solver(
    hess_f: Callable,
    grad_f: Callable,
    x: torch.Tensor,
    w: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    n = int(math.sqrt(x.numel()))
    m = A.shape[0]

    X = x.reshape((n, n))

    grad = grad_f(x)

    g = grad + A.T @ w
    h = A @ x - b

    delta_w = torch.zeros_like(w)

    inv_H_A = (X @ A.reshape((m, n, n)) @ X).permute(1, 2, 0).reshape(-1, m)
    inv_H_g = (X @ g.reshape((n, n)) @ X).ravel()

    S = -A @ inv_H_A

    delta_w = solve(S, A @ inv_H_g - h)

    rhs_x = -A.T @ delta_w - g
    rhs_X = rhs_x.reshape((n, n))

    delta_X = X @ rhs_X @ X
    delta_X = (delta_X + delta_X.T) / 2.0

    delta_x = delta_X.ravel()

    M = rhs_X @ X
    decrement_squared = torch.sum(M * M.T).item()

    return delta_x, delta_w, decrement_squared


def SDP(
    C: torch.Tensor,
    equality_constraints: Sequence[Any],
    X0: torch.Tensor,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
):
    n = X0.shape[0]

    F = lambda x: torch.dot(C.ravel(), x)

    A = []
    b = []

    for A_i, b_i in equality_constraints:
        A.append(A_i.flatten())
        b.append(b_i)

    if A:
        A = torch.stack(A).to(dtype=X0.dtype, device=X0.device)
        b = torch.as_tensor(b, dtype=X0.dtype, device=X0.device)
        equality = (A, b)
    else:
        equality = None

    def psd_inequality(x):
        X = x.reshape((n, n))
        sign, logdet = slogdet(X)

        if sign <= 0:
            return torch.tensor(float("inf"), dtype=x.dtype, device=x.device)

        return -logdet

    G = [(psd_inequality, 1)]

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
    X_star = x_star.reshape((n, n))

    return X_star
