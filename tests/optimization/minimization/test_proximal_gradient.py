import pytest
import numpy as np
import torch

from vicentin.optimization.minimization import proximal_gradient_descent


def soft_thresholding_np(x, gamma, lambda_):
    """Proximal operator for L1 norm: prox_{gamma * lambda * |.|_1}"""
    threshold = gamma * lambda_
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def soft_thresholding_torch(x, gamma, lambda_):
    """Proximal operator for L1 norm (Torch version)"""
    threshold = gamma * lambda_
    return torch.sign(x) * torch.maximum(
        torch.abs(x) - threshold, torch.tensor(0.0).to(x)
    )


def lasso_problem_np(n_samples=50, n_features=20, seed=42):
    """Generates a random Lasso problem (Ax=b) for NumPy."""
    np.random.seed(seed)
    A = np.random.randn(n_samples, n_features)
    true_x = np.random.randn(n_features)
    true_x[np.abs(true_x) < 0.5] = 0  # Make it sparse
    b = A @ true_x + 0.1 * np.random.randn(n_samples)

    lambda_ = 0.5

    def f(x):
        return 0.5 * np.linalg.norm(A @ x - b) ** 2

    def grad_f(x):
        return A.T @ (A @ x - b)

    def g(x):
        return lambda_ * np.linalg.norm(x, ord=1)

    def prox_g(x, gamma):
        return soft_thresholding_np(x, gamma, lambda_)

    return {
        "F": (f, grad_f),
        "G": (g, prox_g),
        "x0": np.zeros(n_features),
        "A": A,
        "b": b,
        "lambda": lambda_,
    }


def lasso_problem_torch(n_samples=50, n_features=20, seed=42):
    """Generates a random Lasso problem for Torch."""
    torch.manual_seed(seed)
    A = torch.randn(n_samples, n_features, dtype=torch.float64)
    true_x = torch.randn(n_features, dtype=torch.float64)
    true_x[torch.abs(true_x) < 0.5] = 0
    b = A @ true_x + 0.1 * torch.randn(n_samples, dtype=torch.float64)

    lambda_ = 0.5

    def f(x):
        return 0.5 * torch.norm(A @ x - b) ** 2

    def g(x):
        return lambda_ * torch.norm(x, p=1)

    def prox_g(x, gamma):
        return soft_thresholding_torch(x, gamma, lambda_)

    return {
        "F": (f,),  # Torch autodiff handles grad
        "G": (g, prox_g),
        "x0": torch.zeros(n_features, dtype=torch.float64),
        "A": A,
        "b": b,
        "lambda": lambda_,
    }


# --- Pytest Fixtures ---


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


def test_convergence_line_search(backend, problem):
    """Test if the algorithm converges using adaptive Backtracking Line Search."""
    F, G, x0 = problem["F"], problem["G"], problem["x0"]

    # Run Solver
    x_sol, loss = proximal_gradient_descent(
        F,
        G,
        x0,
        step_size=None,  # Enable Line Search
        max_iter=200,
        tol=1e-5,
        return_loss=True,
        backend=backend,
    )

    # 1. Check monotonicity: Loss should generally decrease
    assert loss[-1] < loss[0], "Loss did not decrease."

    # 2. Check solution validity (Optimality Condition: 0 \in \partial F(x))
    # For Lasso, we check if x is sparse-ish (not a strict mathematical check, but a sanity check)
    if backend == "torch":
        sparsity = (torch.abs(x_sol) < 1e-4).float().mean()
        assert sparsity > 0, "Lasso solution should induce some sparsity."
    else:
        sparsity = np.mean(np.abs(x_sol) < 1e-4)
        assert sparsity > 0, "Lasso solution should induce some sparsity."


def test_convergence_fixed_step(backend, problem):
    """Test if the algorithm converges using a fixed small step size."""
    F, G, x0 = problem["F"], problem["G"], problem["x0"]

    # Use a conservative step size (1/L)
    # Estimate Lipschitz constant L = ||A^T A||
    if backend == "numpy":
        L = np.linalg.norm(problem["A"].T @ problem["A"], ord=2)
    else:
        L = torch.linalg.norm(problem["A"].T @ problem["A"], ord=2)

    step_size = 1.0 / float(L)

    x_sol, loss = proximal_gradient_descent(
        F,
        G,
        x0,
        step_size=step_size,
        max_iter=200,
        return_loss=True,
        backend=backend,
    )

    assert loss[-1] < loss[0]
    # Fixed step is usually slower, so we just check it didn't diverge
    assert not np.isnan(loss[-1])


def test_zero_regularization_equivalence(backend):
    """
    With g(x) = 0, Proximal Gradient should be identical to Gradient Descent.
    We test this by passing prox_g = identity.
    """
    n = 10
    if backend == "numpy":
        x0 = np.ones(n)
        f = lambda x: np.sum(x**2)
        grad_f = lambda x: 2 * x
        g = lambda x: 0
        prox_g = lambda x, gamma: x  # Identity
        F = (f, grad_f)
        G = (g, prox_g)
    else:
        x0 = torch.ones(n, dtype=torch.float64)
        f = lambda x: torch.sum(x**2)
        g = lambda x: 0.0
        prox_g = lambda x, gamma: x
        F = (f,)
        G = (g, prox_g)

    x_sol = proximal_gradient_descent(
        F, G, x0, max_iter=50, tol=1e-6, backend=backend
    )

    # The minimum of sum(x^2) is 0
    if backend == "numpy":
        assert np.allclose(x_sol, 0, atol=1e-3)
    else:
        assert torch.allclose(x_sol, torch.zeros_like(x_sol), atol=1e-3)


def test_high_regularization_sparsity(backend):
    """
    With extremely high lambda, the solution to Lasso should be exactly zero vector.
    """
    n = 10
    if backend == "numpy":
        x0 = np.random.randn(n)
        lambda_ = 1e5  # Huge regularization
        f = lambda x: np.sum((x - 1) ** 2)  # Centered at 1
        grad_f = lambda x: 2 * (x - 1)
        g = lambda x: lambda_ * np.linalg.norm(x, ord=1)
        prox_g = lambda x, gamma: soft_thresholding_np(x, gamma, lambda_)
        F = (f, grad_f)
        G = (g, prox_g)
    else:
        x0 = torch.randn(n)
        lambda_ = 1e5
        f = lambda x: torch.sum((x - 1) ** 2)
        g = lambda x: lambda_ * torch.norm(x, p=1)
        prox_g = lambda x, gamma: soft_thresholding_torch(x, gamma, lambda_)
        F = (f,)
        G = (g, prox_g)

    x_sol = proximal_gradient_descent(F, G, x0, max_iter=50, backend=backend)

    # Solution should be crushed to zero
    if backend == "numpy":
        assert np.allclose(x_sol, 0, atol=1e-5)
    else:
        assert torch.allclose(x_sol, torch.zeros_like(x_sol), atol=1e-5)


def test_return_options(backend, problem):
    """Test that return_loss toggle works correctly."""
    F, G, x0 = problem["F"], problem["G"], problem["x0"]

    # Case 1: return_loss = False (Default usually returns just x)
    res = proximal_gradient_descent(
        F, G, x0, max_iter=5, return_loss=False, backend=backend
    )
    if backend == "numpy":
        assert isinstance(res, np.ndarray)
    else:
        assert isinstance(res, torch.Tensor)

    # Case 2: return_loss = True
    res_tuple = proximal_gradient_descent(
        F, G, x0, max_iter=5, return_loss=True, backend=backend
    )
    assert isinstance(res_tuple, tuple)
    assert len(res_tuple) == 2
    assert isinstance(res_tuple[1], list)


def test_input_validation():
    """Test that the dispatcher raises errors for invalid inputs."""
    # Using NumPy array but incorrect F tuple length
    x0 = np.array([1.0, 2.0])
    F_bad = (lambda x: x,)  # Missing grad_f
    G = (lambda x: 0, lambda x, g: x)

    with pytest.raises(ValueError, match="NumPy backend requires"):
        proximal_gradient_descent(F_bad, G, x0, backend="numpy")
