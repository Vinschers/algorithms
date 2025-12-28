import pytest
import numpy as np
import torch

from vicentin.optimization.minimization import gradient_descent


class ProblemFactory:
    """
    Standardizes problem creation for Gradient Descent.
    Numpy needs (func, grad). Torch needs just func.
    """

    @staticmethod
    def get_quadratic(backend):
        """
        Problem: f(x) = x^2 + y^2 (Minimum at 0,0)
        """
        if backend == "numpy":

            def f(x):
                return x[0] ** 2 + x[1] ** 2

            def g(x):
                return np.array([2 * x[0], 2 * x[1]])

            return (f, g), np.array([10.0, 10.0])

        elif backend == "pytorch":

            def f(x):
                return x[0] ** 2 + x[1] ** 2

            # Note: requires_grad=True is essential for Torch autodiff
            return f, torch.tensor(
                [10.0, 10.0], dtype=torch.float64, requires_grad=True
            )

    @staticmethod
    def get_rosenbrock(backend, a=1, b=100):
        """
        Problem: Rosenbrock function. Global min at (a, a^2).
        Gradient Descent struggles here, so we use it to test max_iter or adaptive steps.
        """
        if backend == "numpy":

            def f(x):
                return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

            def g(x):
                dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
                dy = 2 * b * (x[1] - x[0] ** 2)
                return np.array([dx, dy])

            return (f, g), np.array([-1.2, 1.0])

        elif backend == "pytorch":

            def f(x):
                return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

            return f, torch.tensor(
                [-1.2, 1.0], dtype=torch.float64, requires_grad=True
            )

    @staticmethod
    def assert_close(val1, val2, backend, atol=1e-3):
        if backend == "numpy":
            np.testing.assert_allclose(val1, val2, atol=atol)
        elif backend == "pytorch":
            if not torch.is_tensor(val1):
                val1 = torch.tensor(val1)
            if not torch.is_tensor(val2):
                val2 = torch.tensor(val2)

            # Auto-cast val2 to match val1's precision (Float vs Double)
            if val1.dtype != val2.dtype:
                val2 = val2.to(dtype=val1.dtype)

            assert torch.allclose(val1, val2, atol=atol)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_quadratic_convergence_adaptive(backend):
    """
    Test convergence on a simple convex function using DEFAULT adaptive line search.
    Should converge to [0,0].
    """
    F, x0 = ProblemFactory.get_quadratic(backend)

    # step_size=None triggers backtracking line search
    res = gradient_descent(
        F, x0, step_size=None, return_loss=False, backend=backend
    )

    ProblemFactory.assert_close(res, [0.0, 0.0], backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_quadratic_convergence_fixed_step(backend):
    """
    Test convergence with a FIXED step size (small enough to be stable).
    """
    F, x0 = ProblemFactory.get_quadratic(backend)

    # step_size=0.1 is safe for x^2 + y^2
    res = gradient_descent(
        F, x0, step_size=0.1, max_iter=200, return_loss=False, backend=backend
    )

    ProblemFactory.assert_close(res, [0.0, 0.0], backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_rosenbrock_hard_problem(backend):
    """
    Rosenbrock is hard for GD. We test that it makes PROGRESS, not necessarily
    that it solves it perfectly in few iterations (unlike Newton).
    """
    F, x0 = ProblemFactory.get_rosenbrock(backend)

    # Give it more iterations and adaptive search
    res, loss_history = gradient_descent(
        F, x0, step_size=None, max_iter=1200, return_loss=True, backend=backend
    )

    # Check that loss actually decreased
    assert loss_history[-1] < loss_history[0]
    # Check we are somewhat close to [1, 1] (loose tolerance for GD)
    ProblemFactory.assert_close(res, [1.0, 1.0], backend, atol=0.2)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_return_loss_flag(backend):
    """
    Verify the return signature changes based on `return_loss`.
    """
    F, x0 = ProblemFactory.get_quadratic(backend)

    # Case 1: return_loss = True
    res1, history = gradient_descent(F, x0, return_loss=True, backend=backend)
    assert isinstance(history, list)
    assert len(history) > 0

    # Case 2: return_loss = False
    res2 = gradient_descent(F, x0, return_loss=False, backend=backend)
    # Result should be just the position (tensor/array), not a tuple
    if backend == "numpy":
        assert isinstance(res2, np.ndarray)
    else:
        assert torch.is_tensor(res2)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_start_at_minimum(backend):
    """
    If we start at the minimum, gradient is 0. It should return immediately.
    """
    if backend == "numpy":
        F = (lambda x: x[0] ** 2, lambda x: np.array([2 * x[0]]))
        x0 = np.array([0.0])
    else:
        F = lambda x: x[0] ** 2
        x0 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

    res, history = gradient_descent(F, x0, return_loss=True, backend=backend)

    # Should finish in 0 or 1 iterations
    assert len(history) <= 2
    ProblemFactory.assert_close(res, [0.0], backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_divergence_check(backend):
    """
    Test behavior with a massive fixed step size on a convex problem.
    It should overshoot and likely explode (result in NaNs or huge numbers),
    but we want to ensure the code doesn't hang.
    """
    F, x0 = ProblemFactory.get_quadratic(backend)

    # Huge step size = unstable
    res, history = gradient_descent(
        F, x0, step_size=100.0, max_iter=10, return_loss=True, backend=backend
    )

    # Loss should typically increase or explode
    assert (
        history[-1] > history[0]
        or np.isnan(history[-1])
        or np.isinf(history[-1])
    )


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_max_iter_limit(backend):
    """
    Ensure the algorithm stops exactly at max_iter if convergence isn't reached.
    """
    F, x0 = ProblemFactory.get_quadratic(backend)

    # Set tiny max_iter
    MAX_ITER = 5
    res, history = gradient_descent(
        F,
        x0,
        max_iter=MAX_ITER,
        tol=1e-50,
        epsilon=1e-50,
        return_loss=True,
        backend=backend,
    )

    # Depending on implementation, history might be max_iter or max_iter+1 (initial point)
    assert len(history) <= MAX_ITER + 2
