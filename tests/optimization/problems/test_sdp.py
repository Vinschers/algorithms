import pytest
import time
import numpy as np
import torch

from vicentin.optimization.problems import SDP


class ProblemFactory:
    """
    Constructs valid SDP problems with known solutions.
    """

    @staticmethod
    def get_min_eigenvalue_problem(backend):
        """
        Problem: Minimize the maximum eigenvalue of a symmetric matrix M.
        Let M = diag(1, 2). We want to find scalar 's' such that M <= s*I.
        This can be cast as an SDP:
            Min s
            s.t. s*I - M >= 0 (Positive Semidefinite)

        However, your solver is in standard form (Min Tr(CX) s.t. Tr(AX)=b, X>=0).
        Standard form conversion is complex, so we test a simpler direct standard form problem.

        Simple Standard Form Problem:
        Min Tr(C X)
        s.t. Tr(X) = 1, X >= 0
        Where C = diag(1, 2).

        The solution should put all "mass" on the smallest diagonal entry of C.
        Optimal X = diag(1, 0) -> Objective = 1*1 + 2*0 = 1.
        """
        if backend == "numpy":
            C = np.diag([1.0, 2.0])
            # Constraint: Tr(I * X) = 1 (i.e., trace of X is 1)
            A1 = np.eye(2)
            b1 = 1.0
            eq_constraints = [(A1, b1)]

            # X0 must be strictly positive definite AND satisfy trace(X)=1
            # X0 = diag(0.5, 0.5) works perfectly.
            X0 = np.diag([0.5, 0.5])

        elif backend == "pytorch":
            C = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
            A1 = torch.eye(2, dtype=torch.float64)
            b1 = 1.0
            eq_constraints = [(A1, b1)]
            X0 = torch.tensor(
                [[0.5, 0.0], [0.0, 0.5]],
                dtype=torch.float64,
                requires_grad=True,
            )

        return C, eq_constraints, X0, 1.0  # Expected objective value

    @staticmethod
    def get_correlation_matrix_problem(backend):
        """
        Find the nearest correlation matrix (diagonal must be 1).
        Problem:
        Min Tr(0*X)  (Feasibility problem really, or minimal norm if we formulated it that way)
        Let's try: Min Tr(X)
        s.t. X_11 = 1, X_22 = 1, X >= 0

        Optimal X should be diag(1, 1). Objective = 2.
        """
        if backend == "numpy":
            C = np.eye(2)  # Min Tr(X)

            # Constraints: X_11 = 1, X_22 = 1
            # A1 selects X_11: [[1,0],[0,0]]
            A1 = np.array([[1.0, 0.0], [0.0, 0.0]])
            b1 = 1.0

            # A2 selects X_22: [[0,0],[0,1]]
            A2 = np.array([[0.0, 0.0], [0.0, 1.0]])
            b2 = 1.0

            eq_constraints = [(A1, b1), (A2, b2)]

            # Start at valid point
            X0 = np.array([[1.0, 0.0], [0.0, 1.0]])

        elif backend == "pytorch":
            C = torch.eye(2, dtype=torch.float64)
            A1 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
            b1 = 1.0
            A2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
            b2 = 1.0

            eq_constraints = [(A1, b1), (A2, b2)]
            X0 = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float64,
                requires_grad=True,
            )

        return C, eq_constraints, X0, [1.0, 1.0]  # Expected Diagonals

    @staticmethod
    def assert_close(val1, val2, backend, atol=1e-3):
        if backend == "numpy":
            np.testing.assert_allclose(val1, val2, atol=atol)
        elif backend == "pytorch":
            if not torch.is_tensor(val1):
                val1 = torch.tensor(val1)
            if not torch.is_tensor(val2):
                val2 = torch.tensor(val2)
            if val1.dtype != val2.dtype:
                val2 = val2.to(dtype=val1.dtype)
            assert torch.allclose(val1, val2, atol=atol)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_trace_minimization(backend):
    """
    Min Tr(CX) s.t. Tr(X)=1, X>=0.
    With C = diag(1, 2).
    Optimal X should have 1.0 at X_11 (associated with smaller cost 1)
    and 0.0 at X_22 (associated with larger cost 2).
    """
    C, constraints, X0, _ = ProblemFactory.get_min_eigenvalue_problem(backend)

    # Run SDP
    X_star = SDP(
        C, constraints, X0, epsilon=1e-5, max_iter=200, backend=backend
    )

    # Check that X_star is close to diag(1, 0)
    expected = [[1.0, 0.0], [0.0, 0.0]]
    ProblemFactory.assert_close(X_star, expected, backend, atol=1e-1)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_correlation_matrix_constraints(backend):
    """
    Test multiple equality constraints fixing the diagonal elements.
    Ensures the solver handles >1 constraint correctly.
    """
    C, constraints, X0, _ = ProblemFactory.get_correlation_matrix_problem(
        backend
    )

    X_star = SDP(C, constraints, X0, epsilon=1e-5, backend=backend)

    # We expect X to remain diagonal diag(1, 1) because C=I prefers minimal trace
    # and constraints force diagonals to be 1.
    expected = [[1.0, 0.0], [0.0, 1.0]]
    ProblemFactory.assert_close(X_star, expected, backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_invalid_initial_point_not_positive_definite(backend):
    """
    Strict Feasibility Check:
    If X0 is not positive definite (e.g. has negative eigenvalues),
    the log-barrier -log(det(X0)) is undefined or complex.
    The solver MUST raise an error.
    """
    C, constraints, _, _ = ProblemFactory.get_min_eigenvalue_problem(backend)

    if backend == "numpy":
        # Create an Indefinite matrix (eigenvalues +1, -1)
        X0_bad = np.array([[1.0, 0.0], [0.0, -1.0]])
    else:
        X0_bad = torch.tensor(
            [[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64, requires_grad=True
        )

    # We expect a ValueError or LinearAlgebraError (domain error for log)
    with pytest.raises((ValueError, RuntimeError, np.linalg.LinAlgError)):
        SDP(C, constraints, X0_bad, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_invalid_constraint_dimensions(backend):
    """
    Test that mismatched dimensions between C and Constraints raise error.
    """
    C, _, X0, _ = ProblemFactory.get_min_eigenvalue_problem(backend)

    if backend == "numpy":
        # Constraint matrix shape (3,3) vs C shape (2,2)
        A_bad = np.eye(3)
        b_bad = 1.0
    else:
        A_bad = torch.eye(3, dtype=torch.float64)
        b_bad = 1.0

    bad_constraints = [(A_bad, b_bad)]

    with pytest.raises((ValueError, RuntimeError)):
        SDP(C, bad_constraints, X0, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_sdp_solver_optimization_check(backend):
    """
    PERFORMANCE BENCHMARK
    ---------------------
    Goal: Verify that the backend uses a structure-exploiting linear solver
    (e.g., avoiding the full O(n^6) Hessian inversion).

    Parameters:
    - n = 40: Creates a Hessian of size 1600x1600.
    - Naive O(n^6): ~4.1 billion ops/iter -> Takes > 2.0s on most CPUs.
    - Optimized O(n^4): -> Takes < 0.1s.

    Expected Result:
    - NumPy: PASS (Fast custom solver).
    - PyTorch: FAIL (Slow generic solver).
    """
    # 1. Setup Large Problem (n=40)
    n = 100
    m = 12
    TIME_LIMIT_SECONDS = 10

    print(f"\n[Backend: {backend}] Benchmarking n={n}...")

    if backend == "numpy":
        np.random.seed(12)

        X0 = np.eye(n)
        C = np.random.randn(n, n)
        C = (C + C.T) / 2
        constraints = []
        for _ in range(m):
            A = np.random.randn(n, n)
            A = (A + A.T) / 2
            b = np.trace(A)  # Ensure feasibility
            constraints.append((A, b))
    else:
        torch.random.manual_seed(12)

        X0 = torch.eye(n, dtype=torch.float64, requires_grad=True)
        C = torch.randn((n, n), dtype=torch.float64)
        C = (C + C.T) / 2
        constraints = []
        for _ in range(m):
            A = torch.randn((n, n), dtype=torch.float64)
            A = (A + A.T) / 2
            b = torch.trace(A)
            constraints.append((A, b))

    # 2. Run the Solver & Measure Time
    start_time = time.time()

    # Run exactly 2 iterations.
    # We ignore the result; we only care about speed.
    SDP(C, constraints, X0, max_iter=2, epsilon=1e-3, backend=backend)

    elapsed = time.time() - start_time
    print(f"  -> Finished in {elapsed:.4f}s")

    # 3. Assert Time Limit
    if elapsed > TIME_LIMIT_SECONDS:
        pytest.fail(
            f"Performance Failure: {backend} took {elapsed:.4f}s, "
            f"exceeding the limit of {TIME_LIMIT_SECONDS}s.\n"
            f"Likely cause: Using O(n^6) naive solver instead of optimized custom solver."
        )
