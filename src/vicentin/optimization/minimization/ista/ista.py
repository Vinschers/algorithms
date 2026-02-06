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
    from .ista_np import ista as ista_np

    dispatcher.register("numpy", ista_np, transform_numpy)
except ModuleNotFoundError:
    pass

try:
    from .ista_torch import ista as ista_torch

    dispatcher.register("torch", ista_torch, transform_autodiff)
except ModuleNotFoundError:
    pass


def ista(
    F: Sequence[Callable] | Callable,
    x0: Any,
    lambda_: float,
    step_size: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = True,
    backend: Optional[str] = None,
):
    """
    Solves the L1-regularized minimization problem using ISTA.

    The objective function is:
    $$ \\min_x F(x) + \\lambda \\|x\\|_1 $$

    where $F(x)$ is a smooth, convex function. The algorithm uses the
    Forward-Backward splitting method (Proximal Gradient Descent).

    Parameters:
    -----------
    F : Sequence[Callable] or Callable
        The smooth part of the objective.
        - NumPy: A tuple `(f, grad_f)`.
        - Torch: A callable `f` (gradients computed via autodiff).
    x0 : Any
        Initial guess for the solution.
    lambda_ : float
        Regularization parameter controlling the sparsity of the solution.
        Must be >= 0.
    step_size : float, optional
        Fixed step size. If None, uses Backtracking Line Search.
    max_iter : int, optional (default=100)
        Maximum number of iterations.
    tol : float, optional (default=1e-6)
        Tolerance for convergence based on function value change.
    epsilon : float, optional (default=1e-8)
        Tolerance for convergence based on proximal gradient norm.
    return_loss : bool, optional (default=True)
        Whether to return the history of objective values.

    Returns:
    --------
    x : Any
        The approximate solution.
    loss : List[float], optional
        The history of objective values $f(x) + \\lambda \\|x\\|_1$.
    """

    dispatcher.detect_backend(x0, backend)
    x0 = dispatcher.cast_values(x0)

    return dispatcher(
        F, x0, lambda_, step_size, max_iter, tol, epsilon, return_loss
    )
