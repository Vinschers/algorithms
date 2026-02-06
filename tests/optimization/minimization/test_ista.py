import pytest
import numpy as np
import torch

from vicentin.optimization.minimization import ista


def lasso_problem_np(n_samples=50, n_features=20, seed=42):
    np.random.seed(seed)
    A = np.random.randn(n_samples, n_features)
    # Create a sparse ground truth
    true_x = np.random.randn(n_features)
    true_x[np.abs(true_x) < 0.8] = 0
    b = A @ true_x + 0.05 * np.random.randn(n_samples)

    # Smooth part f(x) = 0.5 * ||Ax - b||^2
    def f(x):
        return 0.5 * np.linalg.norm(A @ x - b) ** 2

    def grad_f(x):
        return A.T @ (A @ x - b)

    return {"F": (f, grad_f), "x0": np.zeros(n_features), "A": A, "b": b}


def lasso_problem_torch(n_samples=50, n_features=20, seed=42):
    torch.manual_seed(seed)
    A = torch.randn(n_samples, n_features, dtype=torch.float64)
    true_x = torch.randn(n_features, dtype=torch.float64)
    true_x[torch.abs(true_x) < 0.8] = 0
    b = A @ true_x + 0.05 * torch.randn(n_samples, dtype=torch.float64)

    def f(x):
        return 0.5 * torch.norm(A @ x - b) ** 2

    return {
        "F": (f,),
        "x0": torch.zeros(n_features, dtype=torch.float64),
        "A": A,
        "b": b,
    }


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    return request.param


@pytest.fixture
def problem(backend):
    if backend == "numpy":
        return lasso_problem_np()
    else:
        return lasso_problem_torch()


# --- Tests ---


def test_ista_convergence(backend, problem):
    """Test if ISTA converges and reduces the loss."""
    F, x0 = problem["F"], problem["x0"]
    lambda_ = 0.5

    x_sol, loss = ista(
        F, x0, lambda_, max_iter=100, return_loss=True, backend=backend
    )

    assert loss[-1] < loss[0]

    # Check sparsity (soft check)
    if backend == "numpy":
        sparsity = np.mean(np.abs(x_sol) < 1e-4)
    else:
        sparsity = (torch.abs(x_sol) < 1e-4).float().mean().item()

    assert sparsity > 0.0, "Solution should be somewhat sparse for lambda=0.5"


def test_zero_lambda_equivalence(backend):
    """
    If lambda=0, ISTA should behave like Gradient Descent (solution is dense).
    """
    n = 10
    if backend == "numpy":
        x0 = np.ones(n)
        # Simple quadratic: min ||x||^2 -> x=0
        F = (lambda x: np.sum(x**2), lambda x: 2 * x)
    else:
        x0 = torch.ones(n, dtype=torch.float64)
        F = (lambda x: torch.sum(x**2),)

    # Solve with lambda=0 and NO return_loss to test generic return
    x_sol = ista(
        F, x0, lambda_=0.0, max_iter=50, return_loss=False, backend=backend
    )

    if backend == "numpy":
        assert np.allclose(x_sol, 0, atol=1e-3)
    else:
        assert torch.allclose(x_sol, torch.zeros_like(x_sol), atol=1e-3)


def test_high_lambda_sparsity(backend):
    """
    If lambda is huge, the solution should be exactly the zero vector.
    """
    n = 10
    if backend == "numpy":
        x0 = np.random.randn(n)
        # f(x) = (x-1)^2, minimum at 1.
        # But with huge lambda, regularization dominates.
        F = (lambda x: np.sum((x - 1) ** 2), lambda x: 2 * (x - 1))
    else:
        x0 = torch.randn(n, dtype=torch.float64)
        F = (lambda x: torch.sum((x - 1) ** 2),)

    lambda_huge = 1e5

    x_sol = ista(
        F,
        x0,
        lambda_=lambda_huge,
        max_iter=50,
        return_loss=False,
        backend=backend,
    )

    # Must be exactly zero due to soft thresholding
    if backend == "numpy":
        assert np.allclose(x_sol, 0, atol=1e-5)
    else:
        assert torch.allclose(x_sol, torch.zeros_like(x_sol), atol=1e-5)


def test_manual_step_size(backend, problem):
    """Test that manual step size runs without errors."""
    F, x0 = problem["F"], problem["x0"]

    # Very small step size ensures convergence without line search
    x_sol = ista(
        F,
        x0,
        lambda_=0.1,
        step_size=1e-4,
        max_iter=10,
        return_loss=False,
        backend=backend,
    )

    if backend == "numpy":
        assert not np.any(np.isnan(x_sol))
    else:
        assert not torch.any(torch.isnan(x_sol))


def test_input_validation():
    """Ensure proper errors for invalid inputs."""
    x0 = np.zeros(5)
    # Missing gradient for NumPy
    F_bad = (lambda x: x,)

    with pytest.raises(ValueError, match="NumPy backend requires"):
        ista(F_bad, x0, lambda_=0.1, backend="numpy")
