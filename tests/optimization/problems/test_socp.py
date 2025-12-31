import pytest
import numpy as np
import torch
import time


from vicentin.optimization.problems import SOCP


class ProblemFactory:
    """
    Constructs valid SOCP problems with known analytical solutions.
    Focuses on standard geometric interpretations of SOC constraints.
    """

    @staticmethod
    def get_unit_disk_problem(backend: str):
        """
        Problem 1: Linear Minimization on the Unit Disk.
        Minimize f^T x
        s.t. ||x||_2 <= 1  (The unit ball)

        Let f = [-1, -1]. We want to find the point in the unit circle
        furthest in the (1,1) direction.

        Analytical Solution:
        x* = [1/sqrt(2), 1/sqrt(2)] approx [0.7071, 0.7071]

        SOCP Formulation:
        Constraint 1: || I*x + 0 ||_2 <= 0^T*x + 1
          A = I(2), b = [0,0], c = [0,0], d = 1
        """
        if backend == "numpy":
            f = np.array([-1.0, -1.0])

            # ||Ax + b|| <= c^T x + d
            A = np.eye(2)
            b = np.zeros(2)
            c = np.zeros(2)
            d = 1.0

            # x0 = [0, 0] is strictly feasible (0 < 1)
            x0 = np.array([0.0, 0.0])

            constraints = [(A, b, c, d)]

            expected_x = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])

        elif backend == "pytorch":
            dtype = torch.float64
            f = torch.tensor([-1.0, -1.0], dtype=dtype)

            A = torch.eye(2, dtype=dtype)
            b = torch.zeros(2, dtype=dtype)
            c = torch.zeros(2, dtype=dtype)
            d = 1.0

            x0 = torch.tensor([0.0, 0.0], dtype=dtype, requires_grad=True)

            constraints = [(A, b, c, d)]

            expected_x = torch.tensor(
                [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=dtype
            )

        return {"f": f, "socp_constraints": constraints, "x0": x0}, expected_x

    @staticmethod
    def get_norm_minimization_problem(backend: str):
        """
        Problem 2: Geometric Norm Minimization with Equality Constraints.
        Find the point on the line x + y = 4 closest to the origin.

        Formulation:
        Min t
        s.t. || [x, y] ||_2 <= t
             x + y = 4

        Variable vector z = [x, y, t] (size 3)
        Objective f = [0, 0, 1]

        SOC Constraint:
        || [1 0 0; 0 1 0] z + 0 || <= [0 0 1] z + 0

        Analytical Solution:
        Geometric center is (2, 2). Distance is sqrt(8) approx 2.828.
        z* = [2.0, 2.0, 2.8284...]
        """
        if backend == "numpy":
            f = np.array([0.0, 0.0, 1.0])

            # SOC Constraint parts
            A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            b = np.zeros(2)
            c_vec = np.array([0.0, 0.0, 1.0])
            d = 0.0
            soc_constraints = [(A, b, c_vec, d)]

            # Equality Constraint: x + y = 4 -> [1, 1, 0] z = 4
            F = np.array([[1.0, 1.0, 0.0]])
            g = np.array([4.0])

            # Initial Point: Must satisfy Equality AND Strict Cone
            # Try x=2, y=2. ||(2,2)|| = sqrt(8) ~ 2.82
            # Let t = 4.0.   2.82 < 4.0 (Strictly feasible)
            x0 = np.array([2.0, 2.0, 4.0])

            expected_x = np.array([2.0, 2.0, np.sqrt(8)])

        elif backend == "pytorch":
            dtype = torch.float64
            f = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)

            A = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype)
            b = torch.zeros(2, dtype=dtype)
            c_vec = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
            d = 0.0
            soc_constraints = [(A, b, c_vec, d)]

            F = torch.tensor([[1.0, 1.0, 0.0]], dtype=dtype)
            g = torch.tensor([4.0], dtype=dtype)

            x0 = torch.tensor([2.0, 2.0, 4.0], dtype=dtype, requires_grad=True)

            expected_x = torch.tensor([2.0, 2.0, np.sqrt(8)], dtype=dtype)

        return {
            "f": f,
            "socp_constraints": soc_constraints,
            "F": F,
            "g": g,
            "x0": x0,
        }, expected_x

    @staticmethod
    def assert_close(val1, val2, backend, atol=1e-3):
        if backend == "numpy":
            np.testing.assert_allclose(val1, val2, atol=atol)
        elif backend == "pytorch":
            if not torch.is_tensor(val1):
                val1 = torch.tensor(val1)
            if not torch.is_tensor(val2):
                val2 = torch.tensor(val2)
            if val1.requires_grad:
                val1 = val1.detach()
            if val2.requires_grad:
                val2 = val2.detach()
            # Cast to match if necessary
            if val1.dtype != val2.dtype:
                val2 = val2.to(val1.dtype)
            assert torch.allclose(val1, val2, atol=atol)


# ==========================================
# TEST SUITE
# ==========================================


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_socp_unit_disk_minimization(backend):
    """
    Validates basic SOCP functionality: Minimizing a linear objective
    over a simple Quadratic (Ice-cream) cone.
    """
    args, expected_x = ProblemFactory.get_unit_disk_problem(backend)

    # Solve
    x_star = SOCP(**args, max_iter=50, epsilon=1e-5, backend=backend)

    ProblemFactory.assert_close(x_star, expected_x, backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_socp_linear_equality_mixing(backend):
    """
    Validates the solver's ability to handle mixing SOC constraints
    with standard affine equality constraints (Fx = g).
    """
    args, expected_x = ProblemFactory.get_norm_minimization_problem(backend)

    # Solve
    x_star = SOCP(**args, max_iter=50, epsilon=1e-5, backend=backend)

    ProblemFactory.assert_close(x_star, expected_x, backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_socp_infeasible_start(backend):
    """
    Ensures the solver rejects initial points that are not strictly feasible
    (i.e., points on the boundary or outside the cone).
    """
    args, _ = ProblemFactory.get_unit_disk_problem(backend)

    # The cone is ||x|| < 1.
    # Point [1, 0] is on the boundary (not strictly inside).
    # Point [2, 0] is outside.

    if backend == "numpy":
        x0_boundary = np.array([1.0, 0.0])
        x0_outside = np.array([2.0, 0.0])
    else:
        x0_boundary = torch.tensor(
            [1.0, 0.0], dtype=torch.float64, requires_grad=True
        )
        x0_outside = torch.tensor(
            [2.0, 0.0], dtype=torch.float64, requires_grad=True
        )

    # Test boundary rejection
    args["x0"] = x0_boundary
    with pytest.raises((ValueError, RuntimeError)):
        SOCP(**args, backend=backend)

    # Test outside rejection
    args["x0"] = x0_outside
    with pytest.raises((ValueError, RuntimeError)):
        SOCP(**args, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_socp_dimension_mismatch(backend):
    """
    Tests input validation for dimension mismatches between f and A matrices.
    """
    args, _ = ProblemFactory.get_unit_disk_problem(backend)

    # A is 2x2, but we provide f of size 3
    if backend == "numpy":
        args["f"] = np.array([1.0, 1.0, 1.0])
    else:
        args["f"] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    with pytest.raises((ValueError, RuntimeError, IndexError)):
        SOCP(**args, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_socp_performance_benchmark(backend):
    """
    Performance check for a moderate size problem.
    Minimize f^T x s.t. ||x|| <= 10
    Dimensions: n=50
    """
    n = 50
    TIME_LIMIT = 1.0  # seconds

    if backend == "numpy":
        rng = np.random.RandomState(42)
        f = rng.randn(n)
        # Cone: ||I*x + 0|| <= 0*x + 10
        A = np.eye(n)
        b = np.zeros(n)
        c = np.zeros(n)
        d = 10.0
        x0 = np.zeros(n)  # Strictly inside
        constraints = [(A, b, c, d)]
    else:
        torch.manual_seed(42)
        f = torch.randn(n, dtype=torch.float64)
        A = torch.eye(n, dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        c = torch.zeros(n, dtype=torch.float64)
        d = 10.0
        x0 = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        constraints = [(A, b, c, d)]

    start_time = time.time()

    # Run a few iterations
    SOCP(
        f=f,
        socp_constraints=constraints,
        x0=x0,
        max_iter=3,
        epsilon=1e-3,
        backend=backend,
    )

    elapsed = time.time() - start_time
    print(f" -> {backend} finished in {elapsed:.4f}s")

    if elapsed > TIME_LIMIT:
        pytest.fail(f"Performance too slow: {elapsed}s > {TIME_LIMIT}s")
