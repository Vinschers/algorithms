from typing import Callable, Optional
import torch

from vicentin.optimization.minimization import proximal_gradient_descent


def projected_gradient(
    f: Callable,
    projection: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    step_size: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = False,
):
    def g(_):
        return 0.0

    def prox_g(v, _):
        return projection(v)

    F = f
    G = (g, prox_g)

    return proximal_gradient_descent(
        F, G, x0, step_size, max_iter, tol, epsilon, return_loss
    )
