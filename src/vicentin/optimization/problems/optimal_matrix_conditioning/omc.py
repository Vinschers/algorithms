from typing import Any, Optional
from vicentin.utils import Dispatcher

omc = Dispatcher()

try:
    from .omc_np import optimal_matrix_conditioning as omc_np

    omc.register("numpy", omc_np)
except ModuleNotFoundError:
    pass

try:
    from .omc_torch import optimal_matrix_conditioning as omc_torch

    omc.register("torch", omc_torch)
except ModuleNotFoundError:
    pass


def optimal_matrix_conditioning(
    A: Any, epsilon: float = 1e-3, backend: Optional[str] = None
):
    """
    Computes the optimally conditioned matrix $X$ within a spectral neighborhood of $A$.

    This function formulates and solves a Semidefinite Programming (SDP) problem to find
    a symmetric matrix $X$ that minimizes the condition number $\\kappa(X)$, subject to
    the constraint that $X$ remains within a spectral distance $\\epsilon$ of the input
    matrix $A$.

    The optimization problem is defined as:

    $$
    \\begin{aligned}
    & \\underset{X, t}{\\text{minimize}} & & t \\\\
    & \\text{subject to} & & I \\preceq X \\preceq tI \\\\
    & & & \\| X - A \\|_2 \\le \\epsilon \\quad (\\text{i.e., } -\\epsilon I \\preceq X - A \\preceq \\epsilon I)
    \\end{aligned}
    $$

    Where $I \\preceq X \\preceq tI$ normalizes the smallest eigenvalue to at least 1
    and bounds the largest by $t$, effectively minimizing the condition number $t/1$.

    Complexity Analysis:
    -------------------
    - Time Complexity: High. Dominated by the SDP solver. Generally $O(N^4)$ to $O(N^6)$
      depending on the interior-point method used, where $N$ is the matrix dimension.
      Not suitable for very large matrices.
    - Space Complexity: $O(N^2)$ to store the basis matrices and LMI constraints.

    Parameters:
    -----------
    A : numpy.ndarray
        The input square symmetric matrix of shape $(N, N)$.
        Must be symmetric as the function uses `eigvalsh` and constructs a symmetric result.

    epsilon : float, optional (default=1e-3)
        The maximum allowed spectral distance (L2 norm) between the input $A$ and
        the result $X$.
        Constraint: Must satisfy $\\epsilon \\ge 1 - \\lambda_{\\min}(A)$ to ensure the
        feasibility of the constraint $X \\succeq I$. If this is not met, a
        ValueError is raised.

    backend : str, optional (default=None)
        Specifies the computational backend.

    Returns:
    --------
    X : numpy.ndarray
        The optimized symmetric matrix with shape $(N, N)$.
        It is guaranteed to be positive definite with eigenvalues $\\ge 1$.
    """

    omc.detect_backend(A, backend)
    A = omc.cast_values(A)

    return omc(A, epsilon)
