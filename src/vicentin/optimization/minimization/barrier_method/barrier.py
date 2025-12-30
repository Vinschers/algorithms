from typing import Any, Callable, Optional, Sequence

from vicentin.utils import Dispatcher

dispatcher = Dispatcher()


try:
    from .barrier_np import barrier_method as barrier_np

    dispatcher.register("numpy", barrier_np)
except ModuleNotFoundError:
    pass

try:
    from .barrier_torch import barrier_method as barrier_torch

    dispatcher.register("torch", barrier_torch)
except ModuleNotFoundError:
    pass


def barrier_method(
    F: Sequence[Callable] | Callable,
    G: Sequence,
    x0: Any,
    equality: Optional[tuple] = None,
    max_iter: int = 30,
    epsilon: float = 1e-4,
    mu: float = 6,
    linear_solver: Optional[Callable] = None,
    return_dual: bool = False,
    return_loss: bool = False,
    return_t: bool = False,
    backend: Optional[str] = None,
):
    """
    Minimizes a convex function with inequality constraints using the
    Logarithmic Barrier Method.

    This method solves the problem:
    $$
    \\begin{aligned}
    & \\text{minimize} && f_0(x) \\\\
    & \\text{subject to} && f_i(x) \\leq 0, \\; i = 1, \\dots, m \\\\
    & && Ax = b
    \\end{aligned}
    $$

    It approximates the constrained problem by solving a sequence of
    unconstrained "centering" problems. The objective for the centering step is:
    $$ t f_0(x) + \\phi(x) $$
    where $\\phi(x) = -\\sum_{i=1}^m \\log(-f_i(x))$ is the logarithmic barrier.

    The parameter $t$ determines the accuracy of the approximation. The
    algorithm starts with a small $t$, solves the centering problem using
    Newton's Method, increases $t$ by a factor $\\mu$, and repeats until the
    duality gap $m/t$ is less than $\\epsilon$.

    Strict Feasibility:
    -------------------
    - The initial point `x0` must be **strictly feasible** regarding the
      inequality constraints. That is, $f_i(x_0) < 0$ for all $i$.
    - If $f_i(x_0) = 0$ (boundary), the log-barrier is undefined, and the method
      will raise an error.

    Complexity Analysis:
    --------------------
    - The number of outer iterations (centering steps) is generally logarithmic
      in accuracy: $O(\\log(m / \\epsilon))$.
    - The total effort is the sum of Newton steps across all centering steps.
    - Theoretically, for self-concordant functions, the method converges in
      $O(\\sqrt{m} \\log(m/\\epsilon))$ total Newton steps, where $m$ is the
      number of inequality constraints.

    Parameters:
    -----------
    F : Sequence[Callable] or Callable
        The objective function definition.
        - **NumPy:** A tuple `(f, grad_f, hess_f)`.
        - **Torch/JAX:** A single callable `f` (derivatives computed via
          autodiff).
    G : Sequence[Sequence[Callable]] or Sequence[Callable]
        The inequality constraints definition. A sequence where each element
        defines a constraint $f_i$.

        **Constraint Types:**
        - `STANDARD_INEQUALITY` (0): Represents $f_i(x) \\leq 0$.
           Barrier is $-\\log(-f_i(x))$.
        - `LOG_INEQUALITY` (1): Represents a term that is already a barrier
          (e.g. $-\\log x$). The term is added directly to the potential.

        **Input Format:**
        - **NumPy:** A list of tuples. Each tuple is
          `(ineq, grad, hess, [type])`.
          If `type` is omitted, defaults to Standard.
        - **Torch/JAX:** A list where elements are either:
          1. A callable `ineq` (defaults to Standard).
          2. A tuple `(ineq, type)`.
    x0 : Any
        Initial strictly feasible point. Must satisfy $f_i(x_0) < 0$.
        The type  determines the backend usage.
    equality : tuple, optional (default=None)
        Linear equality constraints $(A, b)$ such that $Ax = b$.
    max_iter : int, optional (default=100)
        Maximum number of outer loop iterations (updates to parameter $t$).
    epsilon : float, optional (default=1e-4)
        Convergence tolerance for the duality gap ($m/t < \\epsilon$) to stop
        the outer loop.
    mu : float, optional (default=6)
        Factor by which $t$ is increased at each outer step ($t := \\mu t$).
        Typical values are between 2 and 100.
    linear_solver : Callable, optional (default=None)
        A custom strategy to solve the Newton system, bypassing the default dense
        KKT matrix construction. Use this to exploit problem structure (e.g., in SDP).

        **Signature:**
        `solver(hess_f, grad, x, w, A, b) -> (delta_x, delta_w, decrement_squared)`

        **Inputs:**
        - `hess_f` (Callable): Function to compute Hessian (or Hessian-vector products).
        - `grad` (Array/Tensor): Flattened gradient vector $\\nabla f(x)$.
        - `x` (Array/Tensor): Current primal point (original shape).
        - `w` (Array/Tensor): Current dual variable.
        - `A`, `b`: Constraint parameters.

        **Returns:**
        - `delta_x` (Array/Tensor): Primal update direction (same shape as `x`).
        - `delta_w` (Array/Tensor): Dual update direction.
        - `decrement_squared` (float/0-dim Tensor): The value $\\Delta x^T H \\Delta x$.
          Returning this allows the solver to avoid instantiating the full Hessian $H$.
    return_dual : bool, optional (default=False)
        Whether to return the dual solution of the problem.
    return_loss : bool, optional (default=False)
        Whether to return the sequence of objective values at the end of each
        centering step.
    return_t : bool, optional (default=False)
        Whether to return the last value of `t` from the barrier iteration.
    backend : str, optional (default=None)
        Explicitly specify 'numpy', 'torch', or 'jax'. If None,
        inferred from `x0`.

    Returns:
    --------
    x : Any
        The optimal solution found.
    (lambdas, mu) : Tuple[Any], optional
        The dual solution. `lambdas` represent the inequality multipliers and
        `mu` is the equality multiplier.
    loss : List[float], optional
        The history of objective function values $f_0(x)$. Only returned if
        `return_loss` is True.
    """

    dispatcher.detect_backend(x0, backend)
    x0 = dispatcher.cast_values(x0)

    return dispatcher(
        F,
        G,
        x0,
        equality,
        max_iter,
        epsilon,
        mu,
        linear_solver,
        return_dual,
        return_loss,
        return_t,
    )
