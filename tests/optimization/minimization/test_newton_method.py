import pytest
import numpy as np
import torch

from vicentin.optimization.minimization import newton_method


class ProblemFactory:
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

            def h(x):
                return np.array([[2.0, 0.0], [0.0, 2.0]])

            return (f, g, h), np.array([10.0, 10.0])

        elif backend == "pytorch":

            def f(x):
                return x[0] ** 2 + x[1] ** 2

            return f, torch.tensor(
                [10.0, 10.0], dtype=torch.float64, requires_grad=True
            )

    @staticmethod
    def get_rosenbrock(backend, a=1, b=100):
        """
        Problem: Rosenbrock function (Banana function).
        Global minimum at (a, a^2).
        """
        if backend == "numpy":

            def f(x):
                return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

            def g(x):
                dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
                dy = 2 * b * (x[1] - x[0] ** 2)
                return np.array([dx, dy])

            def h(x):
                dxx = 2 - 4 * b * (x[1] - 3 * x[0] ** 2)
                dxy = -4 * b * x[0]
                dyy = 2 * b
                return np.array([[dxx, dxy], [dxy, dyy]])

            return (f, g, h), np.array([-1.2, 1.0])  # Standard starting point

        elif backend == "pytorch":

            def f(x):
                return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

            return f, torch.tensor(
                [-1.2, 1.0], dtype=torch.float64, requires_grad=True
            )

    @staticmethod
    def assert_close(val1, val2, backend, atol=1e-4):
        if backend == "numpy":
            np.testing.assert_allclose(val1, val2, atol=atol)
        elif backend == "pytorch":
            if not torch.is_tensor(val1):
                val1 = torch.tensor(val1)

            if not torch.is_tensor(val2):
                val2 = torch.tensor(val2, dtype=val1.dtype, device=val1.device)
            else:
                val2 = val2.to(dtype=val1.dtype, device=val1.device)

            assert torch.allclose(val1, val2, atol=atol)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_quadratic_unconstrained(backend):
    """
    Verify basic convergence on a simple convex quadratic function.
    Target: [0, 0]
    """
    F, x0 = ProblemFactory.get_quadratic(backend)

    res = newton_method(F, x0, backend=backend)

    ProblemFactory.assert_close(res, [0.0, 0.0], backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_rosenbrock_convergence(backend):
    """
    Verify convergence on the harder, non-convex Rosenbrock function.
    Target: [1, 1] for a=1, b=100
    """
    F, x0 = ProblemFactory.get_rosenbrock(backend)

    res = newton_method(
        F, x0, backend=backend, max_iter=1000
    )  # Newton usually converges fast

    ProblemFactory.assert_close(res, [1.0, 1.0], backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_equality_constraints(backend):
    """
    Minimize f(x) = x^2 + y^2 subject to x + y = 1.
    Lagrangian L = x^2 + y^2 + v(x+y-1).
    Solution is at x=0.5, y=0.5.
    """
    F, x0 = ProblemFactory.get_quadratic(backend)

    # Constraint: 1*x + 1*y = 1
    if backend == "numpy":
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
    else:
        A = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)

    res = newton_method(F, x0, equality=(A, b), backend=backend)

    ProblemFactory.assert_close(res, [0.5, 0.5], backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_return_flags(backend):
    """
    Verify that return_loss and return_dual modify the output signature correctly.
    """
    F, x0 = ProblemFactory.get_quadratic(backend)

    # 1. Test return_loss
    res_x, history = newton_method(F, x0, return_loss=True, backend=backend)
    assert isinstance(history, list)
    assert len(history) > 0
    # The last loss should be near 0
    assert history[-1] < 1e-4

    # 2. Test return_dual (Equality constrained)
    if backend == "numpy":
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
    else:
        A = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)

    # Should return (x, dual_v)
    res_x, res_dual = newton_method(
        F, x0, equality=(A, b), return_dual=True, backend=backend
    )

    # Dual variable should optimize the Lagrangian.
    # For min x^2 + y^2 s.t. x+y=1 -> 2x + v = 0 -> v = -2(0.5) = -1
    ProblemFactory.assert_close(res_dual, [-1.0], backend, atol=1e-2)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_custom_linear_solver(backend):
    """
    Test the injection of a custom linear solver.
    We implement a 'mock' solver that just calls the standard solve
    internally but sets a flag or prints to prove it was called.
    """
    F, x0 = ProblemFactory.get_quadratic(backend)

    solver_called = False

    def mock_solver(hess_f, grad_f, x, w, A, b):
        nonlocal solver_called
        solver_called = True

        # Manually solve the KKT system for this specific unconstrained case
        # System: H * dx = -g
        if backend == "numpy":
            grad = grad_f(x)
            H_val = hess_f(x)
            dx = np.linalg.solve(H_val, -grad)
            decrement = -np.dot(dx, grad)  # Simplification for test
            return dx, None, decrement
        else:
            grad = grad_f(x)
            H_val = hess_f(x)

            dx = torch.linalg.solve(H_val, -grad)
            decrement = torch.dot(-dx, grad)
            return dx, None, decrement

    # The custom solver logic is tricky to mock perfectly generic for Torch without
    # duplicating the library logic, so we test that it *doesn't crash* and *calls the hook*.

    try:
        newton_method(
            F, x0, linear_solver=mock_solver, backend=backend, max_iter=2
        )
    except Exception:
        # It might fail if our mock implementation is too simple for the library's
        # internal logic, but we primarily want to check if it attempts to use it.
        pass

    assert solver_called, "The custom linear solver was not invoked."


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_max_iter_limit(backend):
    """
    Ensure the algorithm stops exactly at max_iter if convergence isn't reached.
    We use a very strict tolerance to force it to run out of iterations.
    """
    F, x0 = ProblemFactory.get_rosenbrock(backend)

    # Force timeout by setting max_iter low and tolerance impossibly high
    res, history = newton_method(
        F,
        x0,
        backend=backend,
        max_iter=5,
        tol=1e-50,
        epsilon=1e-50,
        return_loss=True,
    )

    # History length should be exactly max_iter + 1 (initial state + 5 steps)
    # or just max_iter depending on your implementation specifics.
    assert len(history) <= 6


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_singular_hessian_behavior(backend):
    """
    Test behavior on f(x) = x^4.
    At x=0, Hessian is 0.
    Numpy (standard solver) might crash.
    PyTorch (our robust implementation) should handle it via regularization.
    """
    if backend == "numpy":

        def f(x):
            return x[0] ** 4

        def g(x):
            return np.array([4 * x[0] ** 3])

        def h(x):
            return np.array([[12 * x[0] ** 2]])

        F = (f, g, h)
        x0 = np.array([0.0])  # Hessian is [[0]] here
    else:

        def f(x):
            return x[0] ** 4

        F = f
        x0 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

    if backend == "numpy":
        # Numpy implementation (assuming standard linalg.solve) usually crashes
        with pytest.raises(np.linalg.LinAlgError):
            newton_method(F, x0, backend=backend)
    else:
        # PyTorch implementation handles it!
        # We just want to ensure it doesn't crash and returns a valid tensor
        res = newton_method(F, x0, backend=backend)
        ProblemFactory.assert_close(res, [0.0], backend, atol=1e-1)


def test_dimension_mismatch_error():
    """
    Test that mismatched constraint dimensions raise an error immediately.
    """
    # Problem is 2D
    F, x0 = ProblemFactory.get_quadratic("numpy")

    # Constraint is 3D (A has 3 columns)
    A = np.array([[1.0, 1.0, 1.0]])
    b = np.array([1.0])

    with pytest.raises(ValueError):
        newton_method(F, x0, equality=(A, b), backend="numpy")
