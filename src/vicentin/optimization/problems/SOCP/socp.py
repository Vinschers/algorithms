from typing import Any, Sequence, Optional

from vicentin.utils import Dispatcher

socp = Dispatcher()

try:
    from .socp_np import SOCP as SOCP_np

    socp.register("numpy", SOCP_np)
except ModuleNotFoundError:
    pass

try:
    from .socp_torch import SOCP as SOCP_torch

    socp.register("torch", SOCP_torch)
except ModuleNotFoundError:
    pass


def SOCP(
    f: Any,
    socp_constraints: Sequence[Sequence[Any]],
    F: Optional[Any] = None,
    g: Optional[Any] = None,
    x0: Any = None,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_dual: bool = False,
    backend: Optional[str] = None,
):
    """
    Solves a Second-Order Cone Program (SOCP) using the Logarithmic Barrier Method.

    This function minimizes a linear objective function subject to a set of
    second-order cone constraints and linear equality constraints. The problem
    is formulated as:

    $$
    \\begin{aligned}
    & \\min_{x} & & f^\\top x \\\\
    & \\text{s.t.} & & \\|A_i x + b_i\\|_2 \\le c_i^\\top x + d_i, \\quad i=1,\\dots,m \\\\
    & & & Fx = g
    \\end{aligned}
    $$

    The second-order cone constraints define a feasible region that is the
    intersection of $m$ "ice cream" cones. These are enforced implicitly by
    transforming them into smooth quadratic constraints and using the
    logarithmic barrier function.



    Implementation Details:
    -----------------------
    - **Constraint Transformation:** To maintain differentiability at the cone
      vertex, the solver typically uses the quadratic difference form:
      $f_i(x) = \\|A_i x + b_i\\|_2^2 - (c_i^\\top x + d_i)^2 \\le 0$.
    - **Backend Dispatch:** Automatically routes to NumPy or PyTorch based on
      the input type of `x0`.
    - **Strict Feasibility:** The starting point `x0` must be strictly inside
      all cones (i.e., $\\|A_i x_0 + b_i\\|_2 < c_i^\\top x_0 + d_i$).

    Complexity Analysis:
    --------------------
    - Dimensionality: Let $n$ be the dimension of the decision vector $x$.
    - Outer Iterations: $O(\\log(m/\\epsilon))$, where $m$ is the number of cones.
    - Per-Step Complexity: Dominated by the Newton system solve, typically
      $O(n^3)$ for dense systems.

    Parameters:
    -----------
    f : Any
        The objective coefficient vector of shape $(n,)$.
    socp_constraints : Sequence[Tuple[Any, Any, Any, Any]]
        A sequence of cone definitions. Each element is a tuple `(A_i, b_i, c_i, d_i)`
        representing the constraint $\\|A_i x + b_i\\|_2 \\le c_i^\\top x + d_i$.
    F : Any, optional
        Linear equality constraint matrix of shape $(p, n)$.
    g : Any, optional
        Linear equality constraint vector of shape $(p,)$.
    x0 : Any
        Initial strictly feasible point of shape $(n,)$.
    max_iter : int, optional (default=100)
        Maximum number of outer loop iterations (barrier parameter updates).
    epsilon : float, optional (default=1e-4)
        Convergence tolerance for the duality gap.
    mu : float, optional (default=6)
        The factor by which the barrier parameter $t$ is increased at each step.
    return_dual : boolean, optional (default=False)
        Whether to return the dual variables $(u_i, v_i)$ for the cones and
        the multiplier $\\nu$ for the equality constraints.
    backend : str, optional (default=None)
        Explicitly specify the backend ('numpy', 'torch').

    Returns:
    --------
    x_star : Any
        The optimal primal solution vector.
    dual_solution : Tuple[List[Tuple[Any, Any]], Any], optional
        Returned only if `return_dual=True`. Contains the list of dual pairs
        $(u_i, v_i)$ for each cone and the equality multiplier $\\nu$.
    """

    socp.detect_backend(x0, backend)
    x0 = socp.cast_values(x0)

    return socp(
        f,
        socp_constraints,
        F,
        g,
        x0,
        max_iter,
        epsilon,
        mu,
        return_dual,
    )
