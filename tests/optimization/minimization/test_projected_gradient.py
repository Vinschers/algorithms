import pytest
import numpy as np
import torch

from vicentin.optimization.minimization import projected_gradient_descent


# --- Problem: Non-Negative Least Squares (min ||Ax-b||^2 s.t. x >= 0) ---


def nnls_problem_np(seed=42):
    np.random.seed(seed)
    A = np.random.randn(20, 10)
    # Construct a true positive solution
    x_true = np.abs(np.random.randn(10))
    b = A @ x_true

    def f(x):
        return 0.5 * np.linalg.norm(A @ x - b) ** 2

    def grad_f(x):
        return A.T @ (A @ x - b)

    def proj(x):
        return np.maximum(x, 0)  # Projection onto non-negative orthant

    return {
        "F": (f, grad_f),
        "proj": proj,
        "x0": np.zeros(10),
        "x_true": x_true,
    }


def nnls_problem_torch(seed=42):
    torch.manual_seed(seed)
    A = torch.randn(20, 10, dtype=torch.float64)
    x_true = torch.abs(torch.randn(10, dtype=torch.float64))
    b = A @ x_true

    def f(x):
        return 0.5 * torch.norm(A @ x - b) ** 2

    def proj(x):
        return torch.clamp(x, min=0.0)

    return {
        "F": (f,),
        "proj": proj,
        "x0": torch.zeros(10, dtype=torch.float64),
        "x_true": x_true,
    }


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    return request.param


@pytest.fixture
def problem(backend):
    return nnls_problem_np() if backend == "numpy" else nnls_problem_torch()


def test_convergence(backend, problem):
    x_sol, loss = projected_gradient_descent(
        problem["F"],
        problem["proj"],
        problem["x0"],
        max_iter=100,
        return_loss=True,
        backend=backend,
    )

    # 1. Check Loss decrease
    assert loss[-1] < loss[0]

    # 2. Check Feasibility (x >= 0)
    if backend == "numpy":
        assert np.all(x_sol >= -1e-6)
    else:
        assert torch.all(x_sol >= -1e-6)


def test_box_constraint(backend):
    """Test constraints: -1 <= x <= 1"""
    n = 5
    # Objective: Minimize ||x - 5||^2 -> unconstrained min is 5.
    # Constrained min should be 1 (boundary).
    if backend == "numpy":
        x0 = np.zeros(n)
        target = 5.0 * np.ones(n)
        F = (lambda x: np.sum((x - target) ** 2), lambda x: 2 * (x - target))
        proj = lambda x: np.clip(x, -1, 1)
        expected = 1.0
    else:
        x0 = torch.zeros(n, dtype=torch.float64)
        target = 5.0 * torch.ones(n, dtype=torch.float64)
        F = (lambda x: torch.sum((x - target) ** 2),)
        proj = lambda x: torch.clamp(x, -1, 1)
        expected = 1.0

    x_sol = projected_gradient_descent(
        F, proj, x0, max_iter=50, return_loss=False, backend=backend
    )

    if backend == "numpy":
        assert np.allclose(x_sol, expected, atol=1e-3)
    else:
        assert torch.allclose(
            x_sol, torch.full_like(x_sol, expected), atol=1e-3
        )
