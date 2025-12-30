import pytest
import time
import numpy as np
import torch

# Assuming the algorithms are in this module
from vicentin.optimization.problems import SDP, SDP_dual


class ProblemFactory:
    """
    Constructs valid SDP problems (Primal and Dual) with known solutions.
    """

    @staticmethod
    def get_min_eigenvalue_problem(backend):
        """
        Problem: Min Tr(C X) s.t. Tr(X)=1, X >= 0
        Where C = diag(1, 2).

        Analytical Solution:
        Primal X*: diag(1, 0) -> Obj = 1.0

        Dual Problem:
        Max b^T y  s.t.  y_1 * A_1 <= C
        Where b = [1], A_1 = I.
        Max y s.t. y * I <= diag(1, 2) => y <= 1.
        Dual y*: [1.0] -> Obj = 1.0
        """
        if backend == "numpy":
            C = np.diag([1.0, 2.0])
            A1 = np.eye(2)
            b1 = 1.0

            # Primal Inputs
            # X0 = diag(0.5, 0.5) satisfies Tr(X)=1 and X > 0
            primal_args = {
                "C": C,
                "equality_constraints": [(A1, b1)],
                "X0": np.diag([0.5, 0.5]),
            }

            # Dual Inputs
            # LMIs structure: [[A1, C]] (Single block)
            # y0 = 0.0 satisfies 0*I < C (Strictly feasible)
            dual_args = {
                "b": np.array([1.0]),
                "LMIs": [[A1, C]],
                "y0": np.array([0.0]),
            }

            expected_X = np.diag([1.0, 0.0])
            expected_y = np.array([1.0])

        elif backend == "pytorch":
            dtype = torch.float64
            C = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=dtype)
            A1 = torch.eye(2, dtype=dtype)

            primal_args = {
                "C": C,
                "equality_constraints": [(A1, 1.0)],
                "X0": torch.tensor(
                    [[0.5, 0.0], [0.0, 0.5]], dtype=dtype, requires_grad=True
                ),
            }

            dual_args = {
                "b": torch.tensor([1.0], dtype=dtype),
                "LMIs": [[A1, C]],
                "y0": torch.tensor([0.0], dtype=dtype, requires_grad=True),
            }

            expected_X = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=dtype)
            expected_y = torch.tensor([1.0], dtype=dtype)

        return primal_args, dual_args, expected_X, expected_y

    @staticmethod
    def get_correlation_matrix_problem(backend):
        """
        Find the nearest correlation matrix (diagonal must be 1).
        Min Tr(I * X) s.t. X_11=1, X_22=1.

        Analytical Solution:
        Primal X*: diag(1, 1) -> Obj = 2.0

        Dual Problem:
        Max y1 + y2
        s.t. y1*A1 + y2*A2 <= I
        diag(y1, y2) <= I  => y1<=1, y2<=1
        Optimal y*: [1.0, 1.0]
        """
        if backend == "numpy":
            C = np.eye(2)  # Min Tr(X)

            # A1 selects X_11: [[1,0],[0,0]]
            A1 = np.array([[1.0, 0.0], [0.0, 0.0]])
            # A2 selects X_22: [[0,0],[0,1]]
            A2 = np.array([[0.0, 0.0], [0.0, 1.0]])

            primal_args = {
                "C": C,
                "equality_constraints": [(A1, 1.0), (A2, 1.0)],
                "X0": np.array([[1.0, 0.0], [0.0, 1.0]]),
            }

            dual_args = {
                "b": np.array([1.0, 1.0]),
                "LMIs": [[A1, A2, C]],  # [A_y1, A_y2, Bound]
                "y0": np.array([0.0, 0.0]),  # 0 < I is strictly feasible
            }

            expected_X = np.eye(2)
            expected_y = np.array([1.0, 1.0])

        elif backend == "pytorch":
            dtype = torch.float64
            C = torch.eye(2, dtype=dtype)
            A1 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=dtype)
            A2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)

            primal_args = {
                "C": C,
                "equality_constraints": [(A1, 1.0), (A2, 1.0)],
                "X0": torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=dtype, requires_grad=True
                ),
            }

            dual_args = {
                "b": torch.tensor([1.0, 1.0], dtype=dtype),
                "LMIs": [[A1, A2, C]],
                "y0": torch.tensor([0.0, 0.0], dtype=dtype, requires_grad=True),
            }

            expected_X = torch.eye(2, dtype=dtype)
            expected_y = torch.tensor([1.0, 1.0], dtype=dtype)

        return primal_args, dual_args, expected_X, expected_y

    @staticmethod
    def assert_close(val1, val2, backend, atol=1e-3):
        if backend == "numpy":
            np.testing.assert_allclose(val1, val2, atol=atol)
        elif backend == "pytorch":
            if not torch.is_tensor(val1):
                val1 = torch.tensor(val1)
            if not torch.is_tensor(val2):
                val2 = torch.tensor(val2)
            # Detach gradients if present for comparison
            if val1.requires_grad:
                val1 = val1.detach()
            if val2.requires_grad:
                val2 = val2.detach()
            # Ensure types match
            if val1.dtype != val2.dtype:
                val2 = val2.to(dtype=val1.dtype)
            assert torch.allclose(val1, val2, atol=atol)


# ==========================================
# 1. ORIGINAL PRIMAL TESTS (Restored)
# ==========================================


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_trace_minimization(backend):
    """
    Min Tr(CX) s.t. Tr(X)=1, X>=0.
    """
    p_args, _, expected_X, _ = ProblemFactory.get_min_eigenvalue_problem(
        backend
    )

    X_star = SDP(**p_args, epsilon=1e-5, max_iter=200, backend=backend)

    ProblemFactory.assert_close(X_star, expected_X, backend, atol=1e-1)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_correlation_matrix_constraints(backend):
    """
    Test multiple equality constraints fixing the diagonal elements.
    """
    p_args, _, expected_X, _ = ProblemFactory.get_correlation_matrix_problem(
        backend
    )

    X_star = SDP(**p_args, epsilon=1e-5, backend=backend)

    ProblemFactory.assert_close(X_star, expected_X, backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_invalid_initial_point_not_positive_definite(backend):
    """
    Strict Feasibility Check: X0 must be positive definite.
    """
    p_args, _, _, _ = ProblemFactory.get_min_eigenvalue_problem(backend)
    C = p_args["C"]
    constraints = p_args["equality_constraints"]

    if backend == "numpy":
        X0_bad = np.array([[1.0, 0.0], [0.0, -1.0]])
    else:
        X0_bad = torch.tensor(
            [[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64, requires_grad=True
        )

    with pytest.raises((ValueError, RuntimeError, np.linalg.LinAlgError)):
        SDP(C, constraints, X0_bad, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_invalid_constraint_dimensions(backend):
    """
    Test that mismatched dimensions between C and Constraints raise error.
    """
    p_args, _, _, _ = ProblemFactory.get_min_eigenvalue_problem(backend)
    C = p_args["C"]
    X0 = p_args["X0"]

    if backend == "numpy":
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
    """
    n = 40
    m = 12
    TIME_LIMIT_SECONDS = 1

    if backend == "numpy":
        rng = np.random.RandomState(12)
        X0 = np.eye(n)
        C = rng.randn(n, n)
        C = (C + C.T) / 2
        constraints = []
        for _ in range(m):
            A = rng.randn(n, n)
            A = (A + A.T) / 2
            b = np.trace(A)
            constraints.append((A, b))
    else:
        torch.manual_seed(12)
        X0 = torch.eye(n, dtype=torch.float64, requires_grad=True)
        C = torch.randn((n, n), dtype=torch.float64)
        C = (C + C.T) / 2
        constraints = []
        for _ in range(m):
            A = torch.randn((n, n), dtype=torch.float64)
            A = (A + A.T) / 2
            b = torch.trace(A)
            constraints.append((A, b))

    start_time = time.time()

    # Run exactly 2 iterations for speed check
    SDP(C, constraints, X0, max_iter=2, epsilon=1e-3, backend=backend)

    elapsed = time.time() - start_time
    print(f"  -> Finished in {elapsed:.4f}s")

    if elapsed > TIME_LIMIT_SECONDS:
        pytest.fail(
            f"Performance Failure: {backend} took {elapsed:.4f}s, "
            f"exceeding the limit of {TIME_LIMIT_SECONDS}s."
        )


# ==========================================
# 2. NEW DUAL & CONSISTENCY TESTS
# ==========================================


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_dual_solver_basic(backend):
    """
    Test the SDP_dual function in isolation against analytical solution.
    """
    _, dual_args, _, expected_y = ProblemFactory.get_min_eigenvalue_problem(
        backend
    )

    y_star = SDP_dual(**dual_args, epsilon=1e-5, backend=backend)

    ProblemFactory.assert_close(y_star, expected_y, backend, atol=1e-2)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_primal_returns_dual_consistency(backend):
    """
    Check if SDP(return_dual=True) returns a 'y' that matches the
    solution found by the explicit SDP_dual solver.
    """
    # Using correlation matrix problem as it has 2 constraints (y is vector size 2)
    p_args, dual_args, _, _ = ProblemFactory.get_correlation_matrix_problem(
        backend
    )

    # 1. Solve Primal asking for Dual -> (X_star, y_implicit)
    X_p, y_p = SDP(**p_args, return_dual=True, epsilon=1e-5, backend=backend)

    # 2. Solve Dual explicitly -> y_explicit
    y_d = SDP_dual(**dual_args, epsilon=1e-5, backend=backend)

    # The y returned by Primal should match the y found by Dual solver
    ProblemFactory.assert_close(y_p, y_d, backend, atol=1e-2)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_dual_returns_primal_consistency(backend):
    """
    Check if SDP_dual(return_dual=True) returns an 'X' that matches the
    solution found by the explicit SDP solver.
    """
    p_args, dual_args, _, _ = ProblemFactory.get_min_eigenvalue_problem(backend)

    # 1. Solve Dual asking for Primal (Lagrange multiplier) -> (y_star, X_implicit)
    y_d, X_d = SDP_dual(
        **dual_args, return_dual=True, epsilon=1e-5, backend=backend
    )

    # 2. Solve Primal explicitly -> X_explicit
    X_p = SDP(**p_args, epsilon=1e-5, backend=backend)

    # The X returned by Dual should match the X found by Primal solver
    ProblemFactory.assert_close(X_d, X_p, backend, atol=1e-2)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_strong_duality_gap(backend):
    """
    Verify Strong Duality: Primal Objective == Dual Objective.
    Primal: min Tr(CX)
    Dual: max b^T y
    """
    p_args, dual_args, _, _ = ProblemFactory.get_min_eigenvalue_problem(backend)

    X_star = SDP(**p_args, epsilon=1e-6, backend=backend)
    y_star = SDP_dual(**dual_args, epsilon=1e-6, backend=backend)

    # Calculate Objectives
    if backend == "numpy":
        primal_obj = np.trace(p_args["C"] @ X_star)
        dual_obj = np.dot(dual_args["b"], y_star)
    else:
        primal_obj = torch.trace(p_args["C"] @ X_star)
        dual_obj = torch.dot(dual_args["b"], y_star)

    # Assert gap is small
    # Note: Primal minimizes, Dual maximizes. At optimality P = D.
    ProblemFactory.assert_close(primal_obj, dual_obj, backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_dual_feasibility_check(backend):
    """
    Ensure the solution returned by SDP_dual satisfies the LMI constraint:
    sum(y_i * A_i) <= C  -->  C - sum(...) >= 0 (PSD)
    """
    _, dual_args, _, _ = ProblemFactory.get_min_eigenvalue_problem(backend)

    y_star = SDP_dual(**dual_args, epsilon=1e-5, backend=backend)

    # Reconstruct the slack matrix S = C - sum(y_i A_i)
    # For this problem: LMIs = [[A1, C]]
    A1 = dual_args["LMIs"][0][0]
    C = dual_args["LMIs"][0][1]

    if backend == "numpy":
        S = C - y_star[0] * A1
        eigvals = np.linalg.eigvalsh(S)
        min_eig = eigvals.min()
        assert min_eig > -1e-4
    else:
        S = C - y_star[0] * A1
        eigvals = torch.linalg.eigvalsh(S)
        min_eig = eigvals.min()
        # detach for assertion
        assert min_eig.detach().item() > -1e-4
