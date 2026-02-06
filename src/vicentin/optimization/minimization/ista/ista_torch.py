from typing import Callable, Optional
import torch

from vicentin.optimization.minimization import proximal_gradient_descent


def soft_thresholding(x: torch.Tensor, threshold: float) -> torch.Tensor:
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, zero)


def ista(
    f: Callable,
    x0: torch.Tensor,
    lambda_: float,
    step_size: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = False,
):
    def g(x):
        return lambda_ * torch.norm(x, p=1)

    def prox_g(x, gamma):
        return soft_thresholding(x, gamma * lambda_)

    F = f
    G = (g, prox_g)

    return proximal_gradient_descent(
        F, G, x0, step_size, max_iter, tol, epsilon, return_loss
    )
