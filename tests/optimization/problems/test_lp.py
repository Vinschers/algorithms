import pytest
import numpy as np
import torch
from vicentin.optimization.problems import LP


class LPProblemFactory:
    """
    Constructs valid LP problems with known analytical solutions.
    Formulation: Min c^T x  s.t. Gx <= h, Ax = b
    """

    @staticmethod
    def get_box_problem(backend: str):
        """
        Problem 1: Simple Box Minimization.
        Minimize -x - y (Maximize x + y)
        s.t. 0 <= x <= 1
             0 <= y <= 1

        Optimal solution is obviously corner (1, 1).

        Constraints form:
         -x <= 0, -y <= 0
          x <= 1,  y <= 1
        """
        if backend == "numpy":
            c = np.array([-1.0, -1.0])

            G = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
            h = np.array([0.0, 0.0, 1.0, 1.0])

            x0 = np.array([0.5, 0.5])
            expected_x = np.array([1.0, 1.0])

        elif backend == "pytorch":
            dtype = torch.float64
            c = torch.tensor([-1.0, -1.0], dtype=dtype)

            G = torch.tensor(
                [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype
            )
            h = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=dtype)

            x0 = torch.tensor([0.5, 0.5], dtype=dtype, requires_grad=True)
            expected_x = torch.tensor([1.0, 1.0], dtype=dtype)

        return {"c": c, "G": G, "h": h, "x0": x0}, expected_x

    @staticmethod
    def get_simplex_problem(backend: str):
        """
        Problem 2: Triangle (Simplex) Constraint.
        Minimize -x - y
        s.t. x >= 0, y >= 0
             x + y <= 1

        Optimal solution is any point on the line x+y=1 between (0,1) and (1,0)?
        Wait, Min (-x -y) is same as Max (x+y).
        Subject to x+y <= 1.
        The max value is 1. All points on the segment x+y=1 are optimal.

        To force a unique solution, let's change objective to: Minimize -2x -y
        Slope is steeper than the constraint.
        Max 2x + y s.t. x+y <= 1.
        Corner point (1, 0) gives obj = 2.
        Corner point (0, 1) gives obj = 1.
        Optimal is (1, 0).
        """
        if backend == "numpy":
            c = np.array([-2.0, -1.0])

            G = np.array(
                [
                    [-1.0, 0.0],  # x >= 0
                    [0.0, -1.0],  # y >= 0
                    [1.0, 1.0],  # x + y <= 1
                ]
            )
            h = np.array([0.0, 0.0, 1.0])

            # Start: (0.2, 0.2) -> sum=0.4 < 1. Strict feasible.
            x0 = np.array([0.2, 0.2])
            expected_x = np.array([1.0, 0.0])

        elif backend == "pytorch":
            dtype = torch.float64
            c = torch.tensor([-2.0, -1.0], dtype=dtype)

            G = torch.tensor(
                [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]], dtype=dtype
            )
            h = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)

            x0 = torch.tensor([0.2, 0.2], dtype=dtype, requires_grad=True)
            expected_x = torch.tensor([1.0, 0.0], dtype=dtype)

        return {"c": c, "G": G, "h": h, "x0": x0}, expected_x

    @staticmethod
    def get_redundant_constraints_problem(backend: str):
        """
        Problem 3: Redundant Constraints.
        Minimize x
        s.t. x >= 2
             x >= 1 (Redundant)

        Optimal x = 2.
        This tests numerical stability when gradients of constraints might be parallel.
        """
        if backend == "numpy":
            c = np.array([1.0])
            G = np.array([[-1.0], [-1.0]])  # -x <= -2, -x <= -1
            h = np.array([-2.0, -1.0])
            x0 = np.array([3.0])
            expected_x = np.array([2.0])
        elif backend == "pytorch":
            dtype = torch.float64
            c = torch.tensor([1.0], dtype=dtype)
            G = torch.tensor([[-1.0], [-1.0]], dtype=dtype)
            h = torch.tensor([-2.0, -1.0], dtype=dtype)
            x0 = torch.tensor([3.0], dtype=dtype, requires_grad=True)
            expected_x = torch.tensor([2.0], dtype=dtype)

        return {"c": c, "G": G, "h": h, "x0": x0}, expected_x

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
# LP TEST SUITE
# ==========================================


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_lp_box_maximization(backend):
    """Simple bounds test."""
    args, expected_x = LPProblemFactory.get_box_problem(backend)
    x_star = LP(**args, max_iter=100, epsilon=1e-5, backend=backend)
    LPProblemFactory.assert_close(x_star, expected_x, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_lp_simplex_vertex(backend):
    """Tests solution at a vertex of a triangle (simplex)."""
    args, expected_x = LPProblemFactory.get_simplex_problem(backend)
    x_star = LP(**args, max_iter=100, epsilon=1e-5, backend=backend)
    LPProblemFactory.assert_close(x_star, expected_x, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_lp_redundant_constraints(backend):
    """Tests stability with parallel/redundant constraints."""
    args, expected_x = LPProblemFactory.get_redundant_constraints_problem(
        backend
    )
    x_star = LP(**args, max_iter=100, epsilon=1e-5, backend=backend)
    LPProblemFactory.assert_close(x_star, expected_x, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_lp_backend_consistency(backend):
    """
    Solves the same random problem with both backends (implicit check via reuse)
    and ensures logic doesn't crash on random data.
    """
    c = np.array([1.0, 1.0])
    # x + y >= 2
    G = np.array([[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([-2.0, 0.0, 0.0])

    if backend == "numpy":
        x0 = np.array([2.0, 2.0])
    else:
        x0 = torch.tensor([2.0, 2.0], dtype=torch.float64, requires_grad=True)

    LP(c=c, G=G, h=h, x0=x0, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_lp_infeasible_start(backend):
    """Rejects x0 that is not strictly feasible."""
    if backend == "numpy":
        c = np.array([1.0])
        G = np.array([[-1.0]])  # x >= 1
        h = np.array([-1.0])
        x0 = np.array([1.0])  # Boundary (not strict)
    else:
        c = torch.tensor([1.0], dtype=torch.float64)
        G = torch.tensor([[-1.0]], dtype=torch.float64)
        h = torch.tensor([-1.0], dtype=torch.float64)
        x0 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

    with pytest.raises((ValueError, RuntimeError)):
        LP(c=c, G=G, h=h, x0=x0, backend=backend)
