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
    max_iter: int = 30,
    tol: float = 1e-8,
    epsilon: float = 1e-4,
    alpha: float = 0.25,
    beta: float = 0.5,
    linear_solver: Optional[Callable] = None,
    return_dual: bool = False,
    return_loss: bool = False,
    backend: Optional[str] = None,
):
    """
    Minimizes a convex scalar function using Newton's Method with Backtracking Line Search.

    The method uses a second-order Taylor approximation of the function $f$ at the
    current point $x$. It computes the Newton step by solving the KKT system:
    $$
    \\begin{bmatrix} \\nabla^2 f(x) & A^\\top \\\\ A & 0 \\end{bmatrix}
    \\begin{bmatrix} \\Delta x \\\\ \\Delta w \\end{bmatrix} =
    \\begin{bmatrix} -\\nabla f(x) - A^\\top w \\\\ b - Ax \\end{bmatrix}
    $$

    The stopping criterion is based on the Newton decrement $\\lambda(x)$, which
    measures the proximity to the optimum in the quadratic norm:
    $$ \\frac{1}{2} \\lambda(x)^2 = \\frac{1}{2} \\Delta x^\\top \\nabla^2 f(x) \\Delta x $$

    The iteration stops when $\\frac{1}{2} \\lambda(x)^2 \\leq \\epsilon$, the absolute
    change in function value falls below `tol`, or `max_iter` is reached.

    Domain and Backtracking:
    -----------------------
    - The objective function $f$ must be defined such that it evaluates to `inf`
      (or raises a domain error) for any $x$ outside its feasible domain.
    - Backtracking line search ensures the Armijo condition (sufficient decrease).

    Complexity Analysis:
    -------------------
    - **Default Solver:** Time Complexity is dominated by the linear system solve.
      If $N$ is the number of variables, solving the dense system is $O(N^3)$.
      *Note:* For matrix problems (like SDP) where $N=n^2$, this becomes $O(n^6)$.
    - **Custom Linear Solver:** Time complexity depends entirely on the implementation
      of `linear_solver`. Structure-exploiting solvers can reduce this significantly
      (e.g., to $O(n^3)$ for SDPs).

    Parameters:
    -----------
    F : Sequence[Callable] or Callable
        The problem definition.
        - For NumPy: A sequence `(f, grad_f, hess_f)`.
        - For Torch/JAX: A single callable `f` (derivatives computed via autodiff).
    x0 : Any
        Initial guess for the minimum. Must be within the domain of $f$.
    equality : tuple, optional (default=None)
        If solving an equality constraint problem of the form A x = b, `equality`
        should be the tuple (A, b).
    max_iter : int, optional (default=100)
        Maximum number of iterations allowed.
    tol : float, optional (default=1e-5)
        Convergence tolerance for the absolute change in function value.
    epsilon : float, optional (default=1e-4)
        Convergence tolerance for the Newton decrement $\\lambda^2 / 2$.
    alpha : float, optional (default=0.25)
        Backtracking parameter (fraction of decrease predicted by gradient).
    beta : float, optional (default=0.5)
        Backtracking step-size reduction factor ($t := \\beta t$).
    linear_solver : Callable, optional (default=None)
        A custom strategy to solve the Newton system, bypassing the default dense
        KKT matrix construction. Use this to exploit problem structure (e.g., in SDP).

        **Signature:**
        `solver(hess_f, grad_f, x, w, A, b) -> (delta_x, delta_w, decrement_squared)`

        **Inputs:**
        - `hess_f` (Callable): Function to compute Hessian (or Hessian-vector products).
        - `grad_f` (Callable): Function to compute gradient vector $\\nabla f(x)$.
        - `x` (Array/Tensor): Current primal point (original shape).
        - `w` (Array/Tensor): Current dual variable.
        - `A`, `b`: Constraint parameters.

        **Returns:**
        - `delta_x` (Array/Tensor): Primal update direction (same shape as `x`).
        - `delta_w` (Array/Tensor): Dual update direction.
        - `decrement_squared` (float/0-dim Tensor): The value $\\Delta x^T H \\Delta x$.
          Returning this allows the solver to avoid instantiating the full Hessian $H$.
    return_dual : bool, optional (default=False)
        Whether to return the dual solution.
    return_loss : bool, optional (default=False)
        Whether to return the sequence of function values $f(x)$ at each iteration.

    Returns:
    --------
    x : Any
        The approximate local minimum.
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
        linear_solver,
        return_dual,
        return_loss,
    )
