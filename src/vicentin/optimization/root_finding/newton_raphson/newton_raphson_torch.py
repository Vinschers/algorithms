from typing import Callable, Any
from warnings import warn

import torch
from torch.func import jacrev


def newton_raphson(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: Any,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = False,
    return_convergence: bool = False,
):
    if torch.is_tensor(x0):
        x = x0.clone().detach().float()
    else:
        x = torch.tensor(x0, dtype=torch.float32)

    if x.ndim == 0:
        x = x.unsqueeze(0)

    loss = []
    converged = False

    grad_f = jacrev(f)

    for i in range(max_iter):
        f_val = f(x)

        if f_val.ndim == 0:
            f_val = f_val.unsqueeze(0)

        if not torch.isfinite(f_val).all():
            warn(f"Divergence detected at iteration {i}.")
            break

        f_norm = torch.linalg.norm(f_val).item()
        loss.append(f_norm)

        if f_norm < tol:
            converged = True
            break

        try:
            jacobian = grad_f(x)
            if isinstance(jacobian, tuple):
                jacobian = jacobian[0]
        except RuntimeError as e:
            warn(f"Autograd failed: {e}")
            break

        if jacobian.shape == (1, 1):
            grad = jacobian.squeeze()
            if torch.abs(grad) < epsilon:
                print("Gradient too close to zero.")
                break
            delta_x = f_val / grad
        else:
            if jacobian.ndim > 2:
                jacobian = jacobian.squeeze()

            result = torch.linalg.solve(jacobian, f_val)
            delta_x = result.solution

        x = x - delta_x.view_as(x)

    if not torch.is_tensor(x0) or x0.ndim == 0:
        x = x.item()

    output = [x]

    if return_loss:
        output.append(loss)

    if return_convergence:
        output.append(converged)

    if len(output) == 1:
        return output[0]

    return output
