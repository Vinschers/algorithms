from typing import Any, Callable, Optional, Sequence

from vicentin.utils import Dispatcher


def transform_numpy(*args, **kwargs):
    if args:
        F = args[0]
        if not isinstance(F, (list, tuple)) or len(F) < 2:
            raise ValueError("NumPy backend requires F=(f, grad_f)")

        f, grad_f, hess_f = F
        new_args = (f, grad_f, hess_f) + args[1:]
        return new_args, kwargs

    F = kwargs.pop("F")
    if not isinstance(F, (list, tuple)) or len(F) < 2:
        raise ValueError("NumPy backend requires F=(f, grad_f)")

    return args, {"f": F[0], "grad_f": F[1], "hess_f": F[2], **kwargs}


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
    from .newton_np import newton as newton_np

    dispatcher.register("numpy", newton_np, transform_numpy)
except ModuleNotFoundError:
    pass

try:
    from .newton_torch import newton as newton_torch

    dispatcher.register("torch", newton_torch, transform_autodiff)
except ModuleNotFoundError:
    pass


def newton(
    F: Sequence[Callable] | Callable,
    x0: Any,
    equality: Optional[tuple] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    epsilon: float = 1e-4,
    alpha: float = 0.25,
    beta: float = 0.5,
    return_dual: bool = False,
    return_loss: bool = False,
    backend: Optional[str] = None,
):
    """
    Minimizes a convex scalar function using Newton's Method with Backtracking Line Search.

    The method uses a second-order Taylor approximation of the function $f$ at the
    current point $x$. It computes the Newton step by solving the system:
    $\\nabla^2 f(x) \\Delta x_{nt} = -\\nabla f(x)$

    The stopping criterion is based on the Newton decrement $\\lambda(x)$, which
    measures the proximity to the optimum in the quadratic norm:
    $\\frac{1}{2} \\lambda(x)^2 = \\frac{1}{2} \\Delta x_{nt}^\\top \\nabla^2 f(x) \\Delta x_{nt}$

    The iteration stops when $\\frac{1}{2} \\lambda(x)^2 \\leq \\epsilon$, the absolute
    change in function value falls below `tol`, or `max_iter` is reached.

    Domain and Backtracking:
    -----------------------
    - The objective function $f$ must be defined such that it evaluates to `inf`
      (or raises a domain error) for any $x$ outside its feasible domain.
    - Backtracking line search ensures the Armijo condition (sufficient decrease):
      $f(x + t \\Delta x_{nt}) \\leq f(x) + \\alpha t \\nabla f(x)^\\top \\Delta x_{nt}$
    - It also ensures that the step size $t$ is small enough so that $x + t \\Delta x_{nt}$
      stays within the domain of $f$.

    Complexity Analysis:
    -------------------
    - Time Complexity: $O(I \\cdot (H + N^3 + L \\cdot F))$ per iteration, where $I$
      is iterations, $H$ is Hessian computation, $N^3$ is solving the linear
      system, $L$ is line search steps, and $F$ is function evaluation.
    - Space Complexity: $O(N^2)$ to store the Hessian matrix.

    Parameters:
    -----------
    F : Sequence[Callable] or Callable
        The problem definition.
        - For NumPy: A sequence `(f, grad_f, hess_f)`.
        - For Torch/JAX: A single callable `f` (derivatives computed via autodiff).
    x0 : Any
        Initial guess for the minimum. Must be within the domain of $f$.
        The type determines the backend (NumPy vs. PyTorch/JAX).
    equality : tuple, optional (default=None)
        If solving an equality constraint problem of the form A x = b, `equality`
        should be the tuple (A, b).
    max_iter : int, optional (default=100)
        Maximum number of iterations allowed.
    tol : float, optional (default=1e-5)
        Convergence tolerance for the absolute change in function value $|f(x^+) - f(x)|$.
    epsilon : float, optional (default=1e-4)
        Convergence tolerance for the Newton decrement $\\lambda^2 / 2$.
    alpha : float, optional (default=0.25)
        Backtracking parameter (fraction of decrease predicted by gradient).
    beta : float, optional (default=0.5)
        Backtracking step-size reduction factor ($t := \\beta t$).
    return_dual : bool, optional (default=False)
        Whether to return the dual solution.
    return_loss : bool, optional (default=False)
        Whether to return the sequence of function values $f(x)$ at each iteration.

    Returns:
    --------
    x : Any
        The approximate local minimum.
    loss : List[float], optional
        Sequence of function values. Only returned if `return_loss` is True.
    """

    dispatcher.detect_backend(x0, backend)
    x0 = dispatcher.cast_values(x0)

    return dispatcher(
        F,
        x0,
        equality,
        max_iter,
        tol,
        epsilon,
        alpha,
        beta,
        return_dual,
        return_loss,
    )
