from typing import Callable, Optional
import numpy as np


from vicentin.optimization.minimization import proximal_gradient_descent


def soft_thresholding(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def ista(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    lambda_: float,
    step_size: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = False,
):

    def g(x):
        return lambda_ * np.linalg.norm(x, ord=1)

    def prox_g(z, gamma):
        return soft_thresholding(z, gamma * lambda_)

    F = (f, grad_f)
    G = (g, prox_g)

    return proximal_gradient_descent(
        F, G, x0, step_size, max_iter, tol, epsilon, return_loss
    )
