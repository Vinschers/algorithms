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
    from .gradient_descent_np import gradient_descent as gradient_descent_np

    dispatcher.register("numpy", gradient_descent_np, transform_numpy)
except ModuleNotFoundError:
    pass

try:
    from .gradient_descent_torch import (
        gradient_descent as gradient_descent_torch,
    )

    dispatcher.register("torch", gradient_descent_torch, transform_autodiff)
except ModuleNotFoundError:
    pass


def gradient_descent(
    F: Sequence[Callable] | Callable,
    x0: Any,
    step_size: Optional[float] = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = True,
    backend: Optional[str] = None,
):
    """
    Minimizes a scalar function using Gradient Descent (possibly with
    Backtracking Line Search).

    The method iteratively updates the solution estimate $x$ by taking a step in
    the direction of the negative gradient. When a step size is not provided,
    it employs a line search to satisfy the Descent Lemma (Armijo condition):

    $f(x_{n+1}) \\leq f(x_n) - \\frac{\\gamma}{2} \\|\\nabla f(x_n)\\|^2$
    $x_{n+1} = x_n - \\gamma \\nabla f(x_n)$

    where $\\gamma$ is the adaptive step size (learning rate). The iteration
    stops when the relative change in function value $|f(x_{n+1}) - f(x_n)|$
    falls below `tol`, the gradient norm $\\|\\nabla f(x)\\|$ falls below
    `epsilon`, or `max_iter` is reached.

    Complexity Analysis:
    -------------------
    - Time Complexity: $O(I \\cdot (G + L \\cdot F))$ per iteration, where $I$
    is the number of iterations, $G$ is gradient computation, $L$ is line search
    steps, and $F$ is function evaluation.
    - Space Complexity: $O(N)$ to store the position and gradient vectors.

    Parameters:
    -----------
    F : Sequence[Callable] or Callable
        The problem definition.
        - For NumPy/Python: A sequence `(f, grad_f)`.
        - For Torch/JAX: A single callable `f` (gradients are computed via
        autodiff).
    x0 : Any
        Initial guess for the minimum. The type of `x0` determines the
        computational backend (NumPy vs. PyTorch).
    step_size : float, optional (default=None)
        If provided, uses a constant step size $\\gamma$. If None, enables
        adaptive backtracking line search.
    max_iter : int, optional (default=100)
        Maximum number of iterations allowed.
    tol : float, optional (default=1e-6)
        Convergence tolerance for the absolute change in function value.
    epsilon : float, optional (default=1e-8)
        Convergence tolerance for the gradient norm $\\|\\nabla f(x)\\|$.
    return_loss : bool, optional (default=False)
        Whether to return the sequence of function values $f(x)$ recorded
        at each iteration.

    Returns:
    --------
    x : Any
        The approximate local minimum. Type matches the input `x0`.
    loss : List[float], optional
        A list of function values at each iteration. Only returned if
        `return_loss` is True.
    """

    dispatcher.detect_backend(x0, backend)
    x0 = dispatcher.cast_values(x0)

    return dispatcher(F, x0, step_size, max_iter, tol, epsilon, return_loss)
