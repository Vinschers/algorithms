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
    from .newton_raphson_np import newton_raphson as newton_raphson_np

    dispatcher.register("numpy", newton_raphson_np, transform_numpy)
except ModuleNotFoundError:
    pass

try:
    from .newton_raphson_torch import newton_raphson as newton_raphson_torch

    dispatcher.register("torch", newton_raphson_torch, transform_autodiff)
except ModuleNotFoundError:
    pass

try:
    from .newton_raphson_jax import newton_raphson as newton_raphson_jax

    dispatcher.register("jax", newton_raphson_jax, transform_autodiff)
except ModuleNotFoundError:
    pass


def newton_raphson(
    F: Sequence[Callable] | Callable,
    x0: Any,
    max_iter: int = 20,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = True,
    return_convergence: bool = True,
    backend: Optional[str] = None,
):
    """
    Solves a system of nonlinear equations using the Newton-Raphson method.

    The method iteratively updates the solution estimate $x$ by solving the
    linear system for the step $\\Delta x$:

    $J_f(x_n) \\Delta x = f(x_n)$
    $x_{n+1} = x_n - \\Delta x$

    where $J_f$ is the Jacobian matrix, gradient, or derivative. The
    iteration stops when the residual norm $\\|f(x)\\|$ falls below `tol` or
    `max_iter` is reached.

    Complexity Analysis:
    -------------------
    - Time Complexity: $O(N^3)$ per iteration (linear system solver).
    - Space Complexity: $O(N^2)$ (Jacobian storage).

    Parameters:
    -----------
    F : Sequence[Callable] or Callable
        The problem definition.
        - For NumPy/Python: A sequence `(f, grad_f)`.
        - For Torch/JAX: A single callable `f` (gradients are computed via autodiff).

        Note on `f` and `grad_f`:
        - Single-variable: `f` returns a scalar, `grad_f` is the derivative $f'(x)$.
        - Multi-variable: `f` returns a scalar, `grad_f` is the gradient $\\nabla f(x)$.
        - Vector-valued: `f` returns a vector (system), `grad_f` is the Jacobian $J(x)$.

    x0 : Any
        Initial guess for the solution. Determines the execution backend.
    max_iter : int, optional (default=100)
        Maximum number of iterations allowed.
    tol : float, optional (default=1e-6)
        Convergence tolerance for the residual norm $\\|f(x)\\|$.
    epsilon : float, optional (default=1e-8)
        Stopping criteria for $\\| \\nabla f(x)\\|$.
    return_loss : bool, optional (default=False)
        Whether to return the losses throughout the iterations.
    return_convergence : bool, optional (default=False)
        Whether to return if the algorithm converged to a solution
    backend : str, optional (default=None)
        Forces a certain backend to compute the algorithm. Can be 'numpy', 'torch', or 'jax'.

    Returns:
    --------
    x : any
        The approximate root of the system. Type matches the input `x0`.
    loss : List[float]
        The residual norm $\\|f(x)\\|$ at each iteration.
    converged : bool
        Whether the algorithm converged
    """

    dispatcher.detect_backend(x0, backend)
    x0 = dispatcher.cast_values(x0)

    return dispatcher(
        F, x0, max_iter, tol, epsilon, return_loss, return_convergence
    )
