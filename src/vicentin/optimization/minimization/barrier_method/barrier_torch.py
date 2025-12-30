from typing import Callable, Optional

import torch
from torch.func import jacrev

from vicentin.optimization.minimization import newton_method

STANDARD_INEQUALITY = 0
LOG_INEQUALITY = 1


def barrier_phi(I: list, x0: torch.Tensor):
    m = len(I)
    ineq_type = [STANDARD_INEQUALITY] * m
    inequalities = []

    for i, ineq in enumerate(I):
        if isinstance(ineq, (list, tuple)):
            ineq_type[i] = ineq[1]
            inequalities.append(ineq[0])
        else:
            inequalities.append(ineq)

    for i in range(m):
        f_i = inequalities[i](x0)

        if ineq_type[i] == STANDARD_INEQUALITY:
            if (f_i.ndim == 0 and f_i >= 0) or (
                f_i.ndim > 0 and torch.any(f_i >= 0)
            ):
                raise ValueError("Initial point is not feasible.")

        elif ineq_type[i] == LOG_INEQUALITY:
            if torch.any(torch.isnan(f_i)) or torch.any(torch.isinf(f_i)):
                raise ValueError("Initial point is not feasible.")

    def phi(x):
        y = 0
        inf = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)

        for i in range(m):
            f = inequalities[i]
            f_x = f(x).view(-1)

            if ineq_type[i] == STANDARD_INEQUALITY:
                if torch.any(f_x >= 0):
                    return inf

                y -= torch.log(-f_x).sum()
            elif ineq_type[i] == LOG_INEQUALITY:
                if torch.any(torch.isnan(f_x)) or torch.any(torch.isinf(f_x)):
                    return inf

                y += f_x.sum()

        return y

    return phi, inequalities, ineq_type


def barrier_method(
    f: Callable,
    g: list,
    x0: torch.Tensor,
    equality: Optional[tuple] = None,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    linear_solver: Optional[Callable] = None,
    return_dual: bool = False,
    return_loss: bool = False,
    return_t: bool = False,
):
    x = x0.clone().detach()
    t = 1
    m = len(g)
    loss = []
    i = 1

    phi, ineqs, ineq_types = barrier_phi(g, x)

    while True:
        F = lambda z: t * f(z) + phi(z)

        x, w = newton_method(
            F, x, equality, linear_solver=linear_solver, return_dual=True
        )
        loss.append(f(x).item())

        duality_gap = m / t
        if duality_gap < epsilon:
            break

        t *= mu
        i += 1

        if i >= max_iter:
            break

    lambdas = []

    for i in range(m):
        lambda_ = 0

        if ineq_types[i] == STANDARD_INEQUALITY:
            f_i = ineqs[i](x)
            lambda_ = -1 / (t * f_i)
        elif ineq_types[i] == LOG_INEQUALITY:
            grad_f_i = jacrev(ineqs[i])(x)
            grad_f_i = (
                grad_f_i[0] if isinstance(grad_f_i, (list, tuple)) else grad_f_i
            )
            lambda_ = -grad_f_i / t

        lambdas.append(lambda_)

    mu = w / t

    output = [x]

    if return_dual:
        output.append((lambdas, mu))

    if return_loss:
        output.append(loss)

    if return_t:
        output.append(t)

    if len(output) == 1:
        return output[0]

    return output
