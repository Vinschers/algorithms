from typing import Any, Callable, Sequence, Tuple

import numpy as np
from numpy.linalg import solve, LinAlgError
from vicentin.optimization.minimization import barrier_method


def cholesky(X):
    return np.linalg.cholesky(X)


def inv(X):
    L = cholesky(X)
    L_inv = solve(L, np.eye(L.shape[0]))
    X_inv = L_inv.T @ L_inv

    return X_inv


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
    return_dual: bool = False,
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
    hess_f = lambda x: 0

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
            L = cholesky(X)
            logdet = 2 * np.sum(np.log(np.diag(L)))
            return -logdet
        except LinAlgError:
            return np.inf

    def psd_grad(x):
        X = x.reshape((n, n))
        return -inv(X).flatten()

    G = [(psd_inequality, psd_grad, None, 1)]

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
    b: np.ndarray | float,
    A: np.ndarray,
    C: np.ndarray,
    y0: np.ndarray,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_dual=False,
):
    b = np.array(b)
    m = b.size
    n = C.shape[0]

    f = lambda y: -b.dot(y)
    grad_f = lambda y: -b
    hess_f = lambda y: np.zeros((m, m))

    F = (f, grad_f, hess_f)

    S = lambda y: C - np.einsum("ijk,i->jk", A, y)

    def psd_inequality(y):
        try:
            L = cholesky(S(y))
            logdet = 2 * np.sum(np.log(np.diag(L)))
            return -logdet
        except LinAlgError:
            return np.inf

    def psd_grad(y):
        S_inv = inv(S(y))
        grad = np.einsum("jk, ikj -> i", S_inv, A)
        return grad

    def psd_hess(y):
        S_inv = inv(S(y))
        M = np.matmul(S_inv, A)
        H = np.einsum("ikl, jlk -> ij", M, M)
        return H

    G = [(psd_inequality, psd_grad, psd_hess, 1)]

    y_star, t = barrier_method(
        F, G, y0, None, max_iter, epsilon, mu, return_t=True
    )

    X_star = inv(S(y_star)) / t

    return (y_star, X_star) if return_dual else y_star


def normalize_to_nested_list(LMIs: Any):
    if isinstance(LMIs, np.ndarray):
        LMIs = [LMIs]

    result = []

    for item in LMIs:
        if isinstance(item, (list, tuple)):
            result.append([np.asanyarray(m) for m in item])

        elif isinstance(item, np.ndarray):
            if item.ndim == 3:
                result.append([mat for mat in item])
            elif item.ndim == 2:
                result.append([item])
            else:
                raise ValueError(f"Array must be 2D or 3D, got {item.ndim}D")

        else:
            raise TypeError(f"Unsupported type in sequence: {type(item)}")

    return result


def block_diag(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.zeros((0, 0))

    shapes = np.array([m.shape for m in matrices])
    out = np.zeros(shapes.sum(axis=0))

    row, col = 0, 0
    for i, (m, n) in enumerate(shapes):
        out[row : row + m, col : col + n] = matrices[i]
        row += m
        col += n

    return out


def SDP_dual(
    b: np.ndarray | float,
    LMIs: Sequence[Sequence[np.ndarray]] | Sequence[np.ndarray] | np.ndarray,
    y0: np.ndarray,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_dual=False,
):
    b = np.atleast_1d(b)
    m = b.size
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

    C = block_diag(c_blocks)

    A = []
    for i in range(m):
        A.append(block_diag(a_stacks[i]))

    A = np.array(A)

    return _single_SDP_dual(b, A, C, y0, max_iter, epsilon, mu, return_dual)
