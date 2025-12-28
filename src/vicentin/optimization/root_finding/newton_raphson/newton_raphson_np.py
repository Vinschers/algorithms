from typing import Callable, Any
from warnings import warn

import numpy as np


def newton_raphson(
    f: Callable[[np.ndarray], np.ndarray],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: Any,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = False,
    return_convergence: bool = False,
):

    x = np.atleast_1d(np.asarray(x0, dtype=float))

    loss = []
    converged = False

    for i in range(max_iter):
        try:
            f_val = np.atleast_1d(f(x))
            f_norm = float(np.linalg.norm(f_val))
        except Exception as e:
            warn(f"Function evaluation failed at iteration {i}: {e}")
            break

        loss.append(f_norm)

        if f_norm < tol:
            converged = True
            break

        if not np.isfinite(f_norm):
            warn(f"Divergence detected at iteration {i}. Loss is {f_norm}.")
            break

        grad = np.asarray(grad_f(x))

        try:
            if grad.ndim > 1:
                delta_x = np.linalg.solve(grad, f_val)
            else:
                if np.abs(grad) < epsilon:
                    warn("Gradient is too close to zero. Stopping.")
                    break
                delta_x = f_val / grad

            x = x - delta_x.reshape(x.shape)

        except np.linalg.LinAlgError:
            warn("Linear Algebra calculation failed (singular matrix?).")
            break

    if np.ndim(x0) == 0:
        x = x[0]

    output = [x]

    if return_loss:
        output.append(loss)

    if return_convergence:
        output.append(converged)

    if len(output) == 1:
        return output[0]

    return output
