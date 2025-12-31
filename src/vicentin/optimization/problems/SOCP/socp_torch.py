from typing import Sequence
import torch
from vicentin.optimization.minimization import barrier_method


def SOCP(
    f: torch.Tensor,
    socp_constraints: Sequence[Sequence[torch.Tensor]],
    F: torch.Tensor,
    g: torch.Tensor,
    x0: torch.Tensor,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_dual: bool = False,
):
    obj = lambda x: f.dot(x)

    if F is None or g is None:
        equality = None
    else:
        equality = (F, g)

    G = []
    for A, b, c, d in socp_constraints:
        constraint = (
            lambda x, A=A, b=b, c=c, d=d: torch.sum((A @ x + b) ** 2)
            - (c.dot(x) + d) ** 2
        )

        upper_cone = lambda x, c=c, d=d: -c.dot(x) - d

        G.append(constraint)
        G.append(upper_cone)

    x_star, (lambdas, mu_vec) = barrier_method(
        obj,
        G,
        x0,
        equality,
        max_iter,
        epsilon,
        mu,
        return_dual=True,
    )

    u = []
    v = []

    for i in range(len(socp_constraints)):
        A, b, c, d = socp_constraints[i]
        l1 = lambdas[2 * i]  # Multiplier for quadratic form
        l2 = lambdas[2 * i + 1]  # Multiplier for (c^\top x + d) >= 0

        u_i = -2 * l1 * (A @ x_star + b)
        v_i = 2 * l1 * (c.dot(x_star) + d) + l2

        u.append(u_i)
        v.append(v_i)

    return (x_star, (u, v, mu_vec)) if return_dual else x_star
