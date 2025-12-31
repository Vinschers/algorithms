import pytest
import numpy as np
import torch
from vicentin.optimization.problems import QP


class QPProblemFactory:
    """
    Constructs valid QP problems with known analytical solutions.
    Formulation: Min (1/2)x'Px + q'x  s.t. Gx <= h, Ax = b
    """

    @staticmethod
    def get_unconstrained_minimum_inside(backend: str):
        """
        Problem 1: Unconstrained minimum lies INSIDE the feasible region.
        Minimize (x-1)^2 + (y-1)^2  s.t.  0 <= x <= 2, 0 <= y <= 2

        Objective expansion:
        f(x,y) = x^2 - 2x + 1 + y^2 - 2y + 1
               = (1/2) [x y] (2I) [x y]' + [-2 -2] [x y]' + const

        P = 2*I, q = [-2, -2]

        Constraints (Box):
         -x <= 0, -y <= 0
          x <= 2,  y <= 2

        Analytical Solution: x* = [1, 1]
        """
        if backend == "numpy":
            P = 2 * np.eye(2)
            q = np.array([-2.0, -2.0])

            # Gx <= h
            G = np.array(
                [
                    [-1.0, 0.0],  # -x <= 0
                    [0.0, -1.0],  # -y <= 0
                    [1.0, 0.0],  # x <= 2
                    [0.0, 1.0],  # y <= 2
                ]
            )
            h = np.array([0.0, 0.0, 2.0, 2.0])

            # Start at center of box (Strictly feasible)
            x0 = np.array([0.5, 0.5])
            expected_x = np.array([1.0, 1.0])

        elif backend == "pytorch":
            dtype = torch.float64
            P = 2 * torch.eye(2, dtype=dtype)
            q = torch.tensor([-2.0, -2.0], dtype=dtype)

            G = torch.tensor(
                [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype
            )
            h = torch.tensor([0.0, 0.0, 2.0, 2.0], dtype=dtype)

            x0 = torch.tensor([0.5, 0.5], dtype=dtype, requires_grad=True)
            expected_x = torch.tensor([1.0, 1.0], dtype=dtype)

        return {"P": P, "q": q, "G": G, "h": h, "x0": x0}, expected_x

    @staticmethod
    def get_active_constraint_problem(backend: str):
        """
        Problem 2: The global minimum is outside, so solution is on boundary.
        Minimize x^2 + y^2  s.t.  x + y >= 2  (or -x -y <= -2)

        Global min is (0,0), but feasible region is half-plane beyond x+y=2.
        Closest point to origin on line x+y=2 is (1, 1).

        P = 2*I, q = [0, 0]
        """
        if backend == "numpy":
            P = 2 * np.eye(2)
            q = np.zeros(2)

            # -x - y <= -2
            G = np.array([[-1.0, -1.0]])
            h = np.array([-2.0])

            # x0 must be strictly feasible: x+y > 2. Try (2, 2) => sum=4
            x0 = np.array([2.0, 2.0])
            expected_x = np.array([1.0, 1.0])

        elif backend == "pytorch":
            dtype = torch.float64
            P = 2 * torch.eye(2, dtype=dtype)
            q = torch.zeros(2, dtype=dtype)

            G = torch.tensor([[-1.0, -1.0]], dtype=dtype)
            h = torch.tensor([-2.0], dtype=dtype)

            x0 = torch.tensor([2.0, 2.0], dtype=dtype, requires_grad=True)
            expected_x = torch.tensor([1.0, 1.0], dtype=dtype)

        return {"P": P, "q": q, "G": G, "h": h, "x0": x0}, expected_x

    @staticmethod
    def get_equality_constrained_problem(backend: str):
        """
        Problem 3: Equality Constraints mixed with Inequalities.
        Minimize x^2 + y^2 + z^2
        s.t.
             x + y + z = 3
             x, y, z >= 0 (Implicitly handled via barrier if we added G,
                           but here we rely on equality forcing feasibility
                           if we start feasible).

        Actually, let's add specific Gx <= h to ensure we test mixing.
        Constraint: z <= 1.5

        Symmetry implies x=y=z=1 is optimal for the sphere.
        z=1 satisfies z<=1.5.

        P = 2*I, q = 0
        """
        if backend == "numpy":
            P = 2 * np.eye(3)
            q = np.zeros(3)

            # Equality: x + y + z = 3
            A = np.array([[1.0, 1.0, 1.0]])
            b = np.array([3.0])

            # Inequality: z <= 1.5 -> [0 0 1]x <= 1.5
            G = np.array([[0.0, 0.0, 1.0]])
            h = np.array([1.5])

            # Start: [1.2, 1.2, 0.6]. Sum=3.0 (Eq ok). z=0.6 < 1.5 (Ineq ok).
            x0 = np.array([1.2, 1.2, 0.6])
            expected_x = np.array([1.0, 1.0, 1.0])

        elif backend == "pytorch":
            dtype = torch.float64
            P = 2 * torch.eye(3, dtype=dtype)
            q = torch.zeros(3, dtype=dtype)

            A = torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype)
            b = torch.tensor([3.0], dtype=dtype)

            G = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
            h = torch.tensor([1.5], dtype=dtype)

            x0 = torch.tensor([1.2, 1.2, 0.6], dtype=dtype, requires_grad=True)
            expected_x = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)

        return {
            "P": P,
            "q": q,
            "G": G,
            "h": h,
            "A": A,
            "b": b,
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
            if val1.dtype != val2.dtype:
                val2 = val2.to(val1.dtype)
            assert torch.allclose(val1, val2, atol=atol)


# ==========================================
# QP TEST SUITE
# ==========================================


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_qp_internal_minimum(backend):
    """Checks if solver finds minimum when it is strictly inside the feasible polytope."""
    args, expected_x = QPProblemFactory.get_unconstrained_minimum_inside(
        backend
    )
    x_star = QP(**args, max_iter=50, epsilon=1e-5, backend=backend)
    QPProblemFactory.assert_close(x_star, expected_x, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_qp_active_boundary(backend):
    """Checks if solver handles constraints that actively cut off the global minimum."""
    args, expected_x = QPProblemFactory.get_active_constraint_problem(backend)
    x_star = QP(**args, max_iter=50, epsilon=1e-5, backend=backend)
    QPProblemFactory.assert_close(x_star, expected_x, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_qp_mixed_constraints(backend):
    """Checks combination of Equality (A) and Inequality (G) constraints."""
    args, expected_x = QPProblemFactory.get_equality_constrained_problem(
        backend
    )
    x_star = QP(**args, max_iter=50, epsilon=1e-5, backend=backend)
    QPProblemFactory.assert_close(x_star, expected_x, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_qp_infeasible_start(backend):
    """Ensures solver rejects x0 that violates strict feasibility (Gx < h)."""
    args, _ = QPProblemFactory.get_active_constraint_problem(backend)

    # The constraint is x + y >= 2 ( -x -y <= -2 )
    # Boundary: (1, 1) -> -2 <= -2 (Not strict)
    # Violation: (0, 0) -> 0 <= -2 (False)

    if backend == "numpy":
        x0_bad = np.array([1.0, 1.0])
    else:
        x0_bad = torch.tensor(
            [1.0, 1.0], dtype=torch.float64, requires_grad=True
        )

    args["x0"] = x0_bad

    with pytest.raises((ValueError, RuntimeError)):
        QP(**args, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_qp_input_validation(backend):
    """Checks dimension mismatch between P and q."""
    args, _ = QPProblemFactory.get_unconstrained_minimum_inside(backend)

    if backend == "numpy":
        args["q"] = np.array([1.0, 1.0, 1.0])  # q size 3, P size 2x2
    else:
        args["q"] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    with pytest.raises((ValueError, RuntimeError, IndexError)):
        QP(**args, backend=backend)
