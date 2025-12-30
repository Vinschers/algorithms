from typing import Any, Callable, Sequence, Tuple
import math

import torch
from torch.linalg import solve
from vicentin.optimization.minimization import barrier_method


def cholesky(X):
    return torch.linalg.cholesky(X)


def inv(X):
    L = cholesky(X)
    L_inv = solve(L, torch.eye(L.shape[0], dtype=L.dtype))
    X_inv = L_inv.T @ L_inv

    return X_inv


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
    return_dual: bool = False,
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

        try:
            L = cholesky(X)
            logdet = 2 * torch.sum(torch.log(torch.diag(L)))
            return -logdet
        except RuntimeError:
            return torch.tensor(float("inf"), dtype=x.dtype, device=x.device)

    G = [(psd_inequality, 1)]

    x_star, (_, y_star) = barrier_method(
        F,
        G,
        X0.flatten(),
        equality,
        max_iter,
        epsilon,
        mu,
        sdp_linear_solver,
        True,
    )

    X_star = x_star.reshape((n, n))
    X_star = (X_star + X_star.T) / 2

    return (X_star, -y_star) if return_dual else X_star


def _single_SDP_dual(
    b: torch.Tensor | float,
    A: torch.Tensor,
    C: torch.Tensor,
    y0: torch.Tensor,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_dual=False,
):
    b = torch.as_tensor(b, device=y0.device, dtype=y0.dtype)
    f = lambda y: -b.dot(y)

    S = lambda y: C - torch.tensordot(A, y, dims=([0], [0]))

    def psd_inequality(y):
        try:
            L = cholesky(S(y))
            logdet = 2 * torch.sum(torch.log(torch.diag(L)))
            return -logdet
        except RuntimeError:
            return torch.tensor(float("inf"), dtype=y.dtype, device=y.device)

    G = [(psd_inequality, 1)]

    y_star, t = barrier_method(
        f, G, y0, None, max_iter, epsilon, mu, return_t=True
    )

    X_star = inv(S(y_star)) / t

    return (y_star, X_star) if return_dual else y_star


def normalize_to_nested_list(LMIs: Any):
    if isinstance(LMIs, torch.Tensor):
        LMIs = [LMIs]

    result = []

    for item in LMIs:
        if isinstance(item, (list, tuple)):
            result.append([torch.as_tensor(m) for m in item])

        elif isinstance(item, torch.Tensor):
            if item.ndim == 3:
                result.append([mat for mat in item])
            elif item.ndim == 2:
                result.append([item])
            else:
                raise ValueError(f"Tensor must be 2D or 3D, got {item.ndim}D")

        else:
            raise TypeError(f"Unsupported type in sequence: {type(item)}")

    return result


def SDP_dual(
    b: torch.Tensor | float,
    LMIs: (
        Sequence[Sequence[torch.Tensor]] | Sequence[torch.Tensor] | torch.Tensor
    ),
    y0: torch.Tensor,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_dual=False,
):
    if isinstance(b, torch.Tensor):
        m = b.numel()
    else:
        m = 1

    LMIs = normalize_to_nested_list(LMIs)

    c_blocks = []
    a_stacks = [[] for _ in range(m)]

    for lmi in LMIs:
        if len(lmi) != m + 1:
            raise ValueError(
                f"LMI block has {len(lmi)} matrices, expected {m} coefficients + 1 constant (total {m+1})."
            )

        coeffs = lmi[:-1]
        constant = lmi[-1]

        c_blocks.append(constant)

        for i in range(m):
            a_stacks[i].append(coeffs[i])

    C = torch.block_diag(*c_blocks)

    A = []
    for i in range(m):
        A.append(torch.block_diag(*a_stacks[i]))

    A = torch.stack(A)

    return _single_SDP_dual(b, A, C, y0, max_iter, epsilon, mu, return_dual)
