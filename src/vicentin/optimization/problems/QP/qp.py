from typing import Any, Optional

from vicentin.utils import Dispatcher

qp = Dispatcher()

try:
    from .qp_np import QP as QP_np

    qp.register("numpy", QP_np)
except ModuleNotFoundError:
    pass

try:
    from .qp_torch import QP as QP_torch

    qp.register("torch", QP_torch)
except ModuleNotFoundError:
    pass


def QP(
    P: Any,
    q: Any,
    G: Optional[Any] = None,
    h: Optional[Any] = None,
    A: Optional[Any] = None,
    b: Optional[Any] = None,
    x0: Any = None,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 10,
    return_dual: bool = False,
    backend: Optional[str] = None,
):
    """
    Solves a Quadratic Program (QP) using the Logarithmic Barrier Method.

    This function minimizes a convex quadratic objective function subject to
    linear inequality and equality constraints. The problem is formulated as:

    $$
    \\begin{aligned}
    & \\min_{x} & & \\frac{1}{2} x^\\top P x + q^\\top x \\\\
    & \\text{s.t.} & & Gx \\le h \\\\
    & & & Ax = b
    \\end{aligned}
    $$

    Implementation Details:
    -----------------------
    - **Backend Dispatch:** Automatically routes to NumPy or PyTorch based on
      the input type of `x0`.
    - **Strict Feasibility:** The starting point `x0` must be strictly feasible
      with respect to the inequality constraints (i.e., $Gx_0 < h$).

    Parameters:
    -----------
    P : Any
        Positive semi-definite matrix of shape $(n, n)$.
    q : Any
        The objective linear coefficient vector of shape $(n,)$.
    G : Any, optional
        Linear inequality constraint matrix of shape $(m, n)$.
    h : Any, optional
        Linear inequality constraint vector of shape $(m,)$.
    A : Any, optional
        Linear equality constraint matrix of shape $(p, n)$.
    b : Any, optional
        Linear equality constraint vector of shape $(p,)$.
    x0 : Any
        Initial strictly feasible point of shape $(n,)$.
    max_iter : int, optional (default=100)
        Maximum number of outer loop iterations (barrier parameter updates).
    epsilon : float, optional (default=1e-4)
        Convergence tolerance for the duality gap.
    mu : float, optional (default=10)
        The factor by which the barrier parameter $t$ is increased at each step.
    return_dual : boolean, optional (default=False)
        Whether to return the dual variables $\\lambda$ for inequalities and
        $\\nu$ for equalities.
    backend : str, optional (default=None)
        Explicitly specify the backend ('numpy', 'torch').

    Returns:
    --------
    x_star : Any
        The optimal primal solution vector.
    dual_solution : Tuple[Any, Any], optional
        Returned only if `return_dual=True`. Contains the tuple (lambda, nu)
        where `lambda` corresponds to $Gx \\le h$ and `nu` to $Ax = b$.
    """

    qp.detect_backend(x0, backend)
    P, q, G, h, A, b, x0 = qp.cast_values(P, q, G, h, A, b, x0)

    return qp(
        P,
        q,
        G,
        h,
        A,
        b,
        x0,
        max_iter,
        epsilon,
        mu,
        return_dual,
    )
