from typing import Callable, Any

import numpy as np


def newton_raphson(
    f: Callable[[np.ndarray], np.ndarray],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: Any,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_loss: bool = False,
):

    x = x0.copy()

    loss = []

    for _ in range(max_iter):
        f_val = np.atleast_1d(f(x))

        if np.isinf(f_val).any():
            raise RuntimeError("Function diverged.")

        f_norm = float(np.linalg.norm(f_val))
        loss.append(f_norm)

        if f_norm < tol:
            break

        grad = grad_f(x)

        if np.any(grad == 0):
            raise RuntimeError("Gradient is zero.")

        if grad.ndim > 1:
            delta_x = np.linalg.solve(grad, f_val)
        else:
            delta_x = f_val / grad

        x = x - delta_x

    return (x, loss) if return_loss else x
