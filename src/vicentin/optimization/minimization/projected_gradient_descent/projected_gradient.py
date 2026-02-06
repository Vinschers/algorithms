from typing import Any, Callable, Optional, Sequence
from vicentin.utils import Dispatcher


def transform_numpy(*args, **kwargs):
    if args:
        F = args[0]
        if not isinstance(F, (list, tuple)) or len(F) < 2:
            raise ValueError("NumPy backend requires F=(f, grad_f)")
        f, grad_f = F
        new_args = (f, grad_f) + args[1:]
        return new_args, kwargs

    F = kwargs.pop("F")
    if not isinstance(F, (list, tuple)) or len(F) < 2:
        raise ValueError("NumPy backend requires F=(f, grad_f)")
    return args, {"f": F[0], "grad_f": F[1], **kwargs}


def transform_autodiff(*args, **kwargs):
    if args:
        F = args[0]
        f = F[0] if isinstance(F, (list, tuple)) else F
        new_args = (f,) + args[1:]
        return new_args, kwargs

    F = kwargs.pop("F")
    f = F[0] if isinstance(F, (list, tuple)) else F
    return args, {"f": f, **kwargs}


dispatcher = Dispatcher()

try:
    from .projected_gradient_np import projected_gradient as pg_np

    dispatcher.register("numpy", pg_np, transform_numpy)
except ModuleNotFoundError:
    pass

try:
    from .projected_gradient_torch import projected_gradient as pg_torch

    dispatcher.register("torch", pg_torch, transform_autodiff)
except ModuleNotFoundError:
    pass


def projected_gradient_descent(
    F: Sequence[Callable] | Callable,
    projection: Callable,
    x0: Any,
    step_size: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = True,
    backend: Optional[str] = None,
):
    """
    Minimizes a function f(x) subject to x in C using Projected Gradient Descent.

    Parameters:
    -----------
    F : Sequence[Callable] | Callable
        The objective function.
    projection : Callable[[x], x]
        A function that maps any vector v to the closest point in the constraint set C.
    x0 : Any
        Initial guess (does not strictly need to be feasible, but helpful).
    """
    dispatcher.detect_backend(x0, backend)
    x0 = dispatcher.cast_values(x0)

    return dispatcher(
        F, projection, x0, step_size, max_iter, tol, epsilon, return_loss
    )
