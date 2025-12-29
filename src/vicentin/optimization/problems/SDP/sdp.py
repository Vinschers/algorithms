from typing import Any, Sequence, Optional

from vicentin.utils import Dispatcher

dispatcher = Dispatcher()


try:
    from .sdp_np import SDP as SDP_np

    dispatcher.register("numpy", SDP_np)
except ModuleNotFoundError:
    pass

try:
    from .sdp_torch import SDP as SDP_torch

    dispatcher.register("torch", SDP_torch)
except ModuleNotFoundError:
    pass


def SDP(
    C: Any,
    equality_constraints: Sequence[Any],
    X0: Any,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    backend: Optional[str] = None,
):
    """
    Solves a Primal Semidefinite Program (SDP) in standard form using the Barrier Method.

    This function minimizes a linear objective function over the intersection of the cone
    of positive semidefinite matrices and an affine subspace. The problem is formulated as:

    $$
    \\begin{aligned}
    & \\min_{X} & & \\text{Tr}(C^\\top X) \\\\
    & \\text{s.t.} & & \\text{Tr}(A_i X) = b_i, \\quad i=1,\\dots,m \\\\
    & & & X \\succeq 0
    \\end{aligned}
    $$

    The inequality constraint $X \\succeq 0$ is enforced implicitly using the logarithmic
    barrier function $\\phi(X) = -\\log(\\det(X))$. The solver iterates through a sequence
    of centered, equality-constrained Newton steps, strictly maintaining primal feasibility
    ($X_k \\succ 0$).

    Implementation Details:
    -----------------------
    - The matrix variable $X$ of shape $(n, n)$ is flattened into a vector of size $n^2$
      for the optimization routine.
    - **NumPy Backend:** Computes gradients and Hessians analytically. The Hessian of
      the barrier is computed as $\\nabla^2 \\phi(X) = X^{-1} \\otimes X^{-1}$ (Kronecker product).
    - **PyTorch Backend:** Uses automatic differentiation (Autograd) to compute derivatives.
    - The barrier parameter $t$ is updated by the factor `mu` at each outer iteration.

    Complexity Analysis:
    --------------------
    - Dimensionality: Let $n$ be the dimension of the matrix $X$ ($X \\in \\mathbb{R}^{n \\times n}$).
    - Time Complexity: This implementation constructs the full Hessian of size $n^2 \\times n^2$.
      Solving the Newton system consequently requires $O((n^2)^3) = O(n^6)$ operations per step.
      *Note: This effectively limits usage to small-scale problems (typically $n \\le 25$).*
    - Space Complexity: $O(n^4)$ to store the dense Hessian matrix.

    Parameters:
    -----------
    C : Any
        The cost matrix of shape $(n, n)$. Defines the linear objective direction.
        The type (NumPy array or Torch Tensor) determines the computational backend.
    equality_constraints : Sequence[Tuple[Any, float]]
        A sequence of linear equality constraints. Each element must be a tuple `(A_i, b_i)`,
        where `A_i` is a matrix of shape $(n, n)$ and `b_i` is a scalar.
        These enforce the condition $\\text{Tr}(A_i X) = b_i$.
    X0 : Any
        Initial strictly feasible point of shape $(n, n)$.
        Must be strictly positive definite ($X_0 \\succ 0$). If equality constraints are
        present, $X_0$ should ideally satisfy them, though the Newton step will project
        deviations back onto the affine subspace.
    max_iter : int, optional (default=100)
        Maximum number of outer loop iterations (barrier parameter updates).
    epsilon : float, optional (default=1e-4)
        Convergence tolerance. The algorithm terminates when the duality gap is less than `epsilon`.
    mu : float, optional (default=6)
        The factor by which the barrier parameter $t$ is increased at each outer step.
        Controls the aggressiveness of the central path traversal.
    backend : str, optional (default=None)
        Explicitly specify the backend ('numpy', 'torch'). If None, it is inferred
        automatically from the type of `X0`.

    Returns:
    --------
    X_star : Any
        The optimal positive semidefinite matrix $X^*$ that minimizes the objective.
        Returns a NumPy array or Torch Tensor matching the input type.
    """

    dispatcher.detect_backend(X0, backend)
    X0 = dispatcher.cast_values(X0)

    return dispatcher(C, equality_constraints, X0, max_iter, epsilon, mu)
