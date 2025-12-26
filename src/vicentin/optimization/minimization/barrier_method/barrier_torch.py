from typing import Callable

import torch

from vicentin.optimization.minimization import newton_method


def barrier_phi(inequalities: list[Callable]):
    def phi(x):
        y_arr = torch.cat([f(x).view(-1) for f in inequalities])

        if torch.any(y_arr >= 0):
            return torch.tensor(float("inf"), dtype=x.dtype, device=x.device)

        return -torch.log(-y_arr).sum()

    return phi


def barrier_method(
    f: Callable,
    inequalities: list[Callable],
    x0: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-5,
    epsilon: float = 1e-4,
    mu: float = 36,
    return_loss: bool = False,
):
    phi = barrier_phi(inequalities)

    x = x0.clone().detach().to(torch.float64)
    t = 1
    m = len(inequalities)
    loss = []
    i = 1

    while True:
        F = lambda z: t * f(z) + phi(z)

        x = newton_method(F, x)
        f_x = f(x)

        loss.append(f(x))
        if torch.abs(f_x) < tol:
            break

        duality_gap = m / t
        if duality_gap < epsilon:
            break

        t *= mu
        i += 1

        if i >= max_iter:
            break

    return (x, loss) if return_loss else x
