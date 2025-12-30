from typing import Any, Sequence, Optional

from vicentin.utils import Dispatcher

sdp = Dispatcher()
sdp_dual = Dispatcher()


try:
    from .sdp_np import SDP as SDP_np

    sdp.register("numpy", SDP_np)
except ModuleNotFoundError:
    pass

try:
    from .sdp_torch import SDP as SDP_torch

    sdp.register("torch", SDP_torch)
except ModuleNotFoundError:
    pass


def SDP(
    C: Any,
    equality_constraints: Sequence[Any],
    X0: Any,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_dual=False,
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
    return_dual : boolean, optional (default=False)
        Whether to return the solution to the dual SDP problem
    backend : str, optional (default=None)
        Explicitly specify the backend ('numpy', 'torch'). If None, it is inferred
        automatically from the type of `X0`.

    Returns:
    --------
    X_star : Any
        The optimal positive semidefinite matrix $X^*$ that minimizes the objective.
        Returns a NumPy array or Torch Tensor matching the input type.
    """

    sdp.detect_backend(X0, backend)
    X0 = sdp.cast_values(X0)

    return sdp(C, equality_constraints, X0, max_iter, epsilon, mu, return_dual)


try:
    from .sdp_np import SDP_dual as SDP_dual_np

    sdp_dual.register("numpy", SDP_dual_np)
except ModuleNotFoundError:
    pass

try:
    from .sdp_torch import SDP_dual as SDP_dual_torch

    sdp_dual.register("torch", SDP_dual_torch)
except ModuleNotFoundError:
    pass


def SDP_dual(
    b: Any,
    LMIs: Sequence[Sequence[Any]] | Sequence[Any] | Any,
    y0: Any,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_dual=False,
    backend: Optional[str] = None,
):
    """
    Solves a Dual Semidefinite Program (SDP) using the Barrier Method.

    This function maximizes a linear objective function subject to Linear Matrix Inequality (LMI)
    constraints. The problem is formulated in the standard dual form:

    $$
    \\begin{aligned}
    & \\max_{y} & & b^\\top y \\\\
    & \\text{s.t.} & & \\sum_{i=1}^m y_i A_i^{(k)} \\preceq C^{(k)}, \\quad \\forall k
    \\end{aligned}
    $$

    where $y \\in \\mathbb{R}^m$ is the optimization variable, and the constraints are enforced
    over $k$ distinct blocks (or a single aggregated block). The algorithm minimizes the
    equivalent objective $f(y) = -b^\\top y$ while maintaining the strict feasibility of the
    barrier function $\\phi(y) = -\\sum_k \\log(\\det(S^{(k)}(y)))$, where
    $S^{(k)}(y) = C^{(k)} - \\sum_i y_i A_i^{(k)}$.

    Implementation Details:
    -----------------------
    - **LMI Handling:** Multiple LMI blocks are aggregated into a single large block-diagonal
      system $S(y) = \\text{diag}(S^{(1)}(y), \\dots, S^{(k)}(y))$.

    Complexity Analysis:
    --------------------
    - Dimensionality: Let $m$ be the number of variables (length of $b$) and $N$ be the
      total size of the aggregated LMI matrix ($N = \\sum \\text{size}(C^{(k)})$).
    - Time Complexity: The dominant cost per iteration is constructing the Hessian and
      solving the Newton system.
      - Hessian Construction: $O(m N^3 + m^2 N^2)$ operations (matrix inversions and contractions).
      - Linear Solve: $O(m^3)$ to solve for the Newton step.
      *Note: This implementation is best suited for problems where $N$ is small to moderate
      (e.g., $N \\le 100$), as the $O(N^3)$ inversion cost accumulates quickly.*
    - Space Complexity: $O(N^2)$ to store the barrier matrix and $O(m^2)$ for the Hessian.

    Parameters:
    -----------
    b : Union[np.ndarray, float]
        The objective vector of size $m$. Defines the direction to maximize ($b^\\top y$).
    LMIs : Sequence[Sequence[Any]]
        A structured list of LMI blocks. Each block is a sequence containing $m+1$ matrices:
        `[A_1, A_2, ..., A_m, C]`.
        - The first $m$ matrices correspond to the coefficients of $y_i$.
        - The last matrix is the constant bound $C$.
        - All matrices in a block must have the same shape $(n_k, n_k)$.
    y0 : Any
        Initial strictly feasible point of shape $(m,)$.
        Must satisfy the condition $\\sum y_i A_i < C$ (strict inequality) for all blocks,
        ensuring the initial slack matrix is strictly positive definite.
    max_iter : int, optional (default=100)
        Maximum number of outer loop iterations (barrier parameter updates).
    epsilon : float, optional (default=1e-4)
        Convergence tolerance. The algorithm terminates when the duality gap is less than `epsilon`.
    mu : float, optional (default=6)
        The factor by which the barrier parameter is increased at each outer step.
    return_dual : boolean, optional (default=False)
        If True, returns a tuple `(y_star, X_star)`, where `X_star` is the optimal solution
        to the corresponding Primal SDP problem (the Lagrange multiplier of the LMI constraint).
    backend : str, optional (default=None)
        Explicitly specify the backend ('numpy', 'torch'). If None, inferred from input types.

    Returns:
    --------
    y_star : Any
        The optimal vector $y^*$ that maximizes $b^\\top y$. Returns a NumPy array or
        Torch Tensor matching the input type.
    X_star : Any
        The optimal solution to the corresponding Primal SDP problem (the Lagrange multiplier).
    """

    sdp_dual.detect_backend(y0, backend)
    b, LMIs, y0 = sdp_dual.cast_values(b, LMIs, y0)

    return sdp_dual(b, LMIs, y0, max_iter, epsilon, mu, return_dual)
