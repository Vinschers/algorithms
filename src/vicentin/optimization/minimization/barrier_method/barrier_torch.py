from typing import Callable, Optional

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
    equality: Optional[tuple] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_loss: bool = False,
):
    phi = barrier_phi(inequalities)

    x = x0.clone().detach().to(torch.float64)
    t = 1
    m = len(inequalities)
    loss = []
    i = 1

    for f_i in inequalities:
        if f_i(x) >= 0:
            raise ValueError("Initial point is not feasible.")

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
