from typing import Callable, Optional

import torch

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

                y -= f_x.sum()

        return y

    return phi


def barrier_method(
    f: Callable,
    inequalities: list,
    x0: torch.Tensor,
    equality: Optional[tuple] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_loss: bool = False,
):
    x = x0.clone().detach()
    t = 1
    m = len(inequalities)
    loss = []
    i = 1

    phi = barrier_phi(inequalities, x)

    while True:
        F = lambda z: t * f(z) + phi(z)

        y = f(x).item()
        x = newton_method(F, x, equality)
        y_new = f(x).item()

        loss.append(y_new)
        if abs(y_new - y) < tol:
            break

        duality_gap = m / t
        if duality_gap < epsilon:
            break

        t *= mu
        i += 1

        if i >= max_iter:
            break

    return (x, loss) if return_loss else x
