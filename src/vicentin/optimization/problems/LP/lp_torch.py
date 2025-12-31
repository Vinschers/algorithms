import torch
from vicentin.optimization.minimization import barrier_method


def LP(
    c: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 10,
    return_dual: bool = False,
):
    obj = lambda x: c.dot(x)

    if A is None or b is None:
        equality = None
    else:
        equality = (A, b)

    constraints = []

    if G is not None and h is not None:
        for i in range(len(h)):
            c_func = lambda x, g_row=G[i], h_val=h[i]: g_row.dot(x) - h_val
            constraints.append(c_func)

    x_star, (lambdas, nu_vec) = barrier_method(
        obj,
        constraints,
        x0,
        equality,
        max_iter,
        epsilon,
        mu,
        return_dual=True,
    )

    return (x_star, (lambdas, nu_vec)) if return_dual else x_star
