import pytest
import numpy as np
import torch

from vicentin.optimization.minimization import barrier_method


class ProblemFactory:
    """
    Standardizes problem creation for the Barrier Method.
    Numpy requires explicit derivatives for F and G.
    Torch uses autodiff.
    """

    @staticmethod
    def get_1d_linear(backend):
        """
        Problem: Minimize f(x) = x
        Subject to: x >= 1  =>  1 - x <= 0
        Optimum: x = 1
        """
        if backend == "numpy":
            # Objective: f(x) = x
            F = (
                lambda x: x[0],  # f
                lambda x: np.array([1.0]),  # grad
                lambda x: np.array([[0.0]]),  # hess
            )
            # Inequality: 1 - x <= 0
            G = [
                (
                    lambda x: 1.0 - x[0],  # g
                    lambda x: np.array([-1.0]),  # grad
                    lambda x: np.array([[0.0]]),  # hess
                )
            ]
            x0 = np.array([2.0])  # Strictly feasible (2 > 1)

        elif backend == "pytorch":
            F = lambda x: x[0]
            # Inequality: 1 - x <= 0
            G = [lambda x: 1.0 - x[0]]
            x0 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        return F, G, x0

    @staticmethod
    def get_2d_quadratic(backend):
        """
        Problem: Minimize f(x) = x^2 + y^2
        Subject to: x + y >= 2  =>  2 - x - y <= 0
        Optimum: (1, 1)
        """
        if backend == "numpy":
            # Objective: x^2 + y^2
            F = (
                lambda x: x[0] ** 2 + x[1] ** 2,
                lambda x: np.array([2 * x[0], 2 * x[1]]),
                lambda x: np.array([[2.0, 0.0], [0.0, 2.0]]),
            )
            # Inequality: 2 - x - y <= 0
            G = [
                (
                    lambda x: 2.0 - x[0] - x[1],
                    lambda x: np.array([-1.0, -1.0]),
                    lambda x: np.zeros((2, 2)),
                )
            ]
            x0 = np.array([2.0, 2.0])  # Strictly feasible (2+2 >= 2)

        elif backend == "pytorch":
            F = lambda x: x[0] ** 2 + x[1] ** 2
            G = [lambda x: 2.0 - x[0] - x[1]]
            x0 = torch.tensor(
                [2.0, 2.0], dtype=torch.float64, requires_grad=True
            )

        return F, G, x0

    @staticmethod
    def assert_close(val1, val2, backend, atol=1e-3):
        if backend == "numpy":
            np.testing.assert_allclose(val1, val2, atol=atol)
        elif backend == "pytorch":
            if not torch.is_tensor(val1):
                val1 = torch.tensor(val1)
            if not torch.is_tensor(val2):
                val2 = torch.tensor(val2)

            # Auto-cast
            if val1.dtype != val2.dtype:
                val2 = val2.to(dtype=val1.dtype)

            assert torch.allclose(val1, val2, atol=atol)


# ==========================================
# 🧪 Test Cases
# ==========================================


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_1d_linear_boundary(backend):
    """
    Test simple boundary convergence.
    Min x s.t. x >= 1. Should stop exactly at 1.
    """
    F, G, x0 = ProblemFactory.get_1d_linear(backend)

    # We use a tighter epsilon to ensure it gets close to the barrier
    res = barrier_method(F, G, x0, epsilon=1e-5, backend=backend)

    ProblemFactory.assert_close(res, [1.0], backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_2d_quadratic_inequality(backend):
    """
    Test minimization with a linear inequality constraint.
    Min x^2 + y^2 s.t. x + y >= 2.
    """
    F, G, x0 = ProblemFactory.get_2d_quadratic(backend)

    res = barrier_method(F, G, x0, epsilon=1e-5, backend=backend)

    ProblemFactory.assert_close(res, [1.0, 1.0], backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_equality_and_inequality(backend):
    """
    Test combining Equality and Inequality.
    Problem: Min x^2 + y^2
    Subject to:
       1. x >= 0.5  (Inequality: 0.5 - x <= 0)
       2. y = 1.0   (Equality)
    Optimum: (0.5, 1.0)
    """
    # Setup Objective and Inequality (x >= 0.5)
    if backend == "numpy":
        F = (
            lambda x: x[0] ** 2 + x[1] ** 2,
            lambda x: np.array([2 * x[0], 2 * x[1]]),
            lambda x: np.array([[2.0, 0.0], [0.0, 2.0]]),
        )
        G = [
            (
                lambda x: 0.5 - x[0],
                lambda x: np.array([-1.0, 0.0]),
                lambda x: np.zeros((2, 2)),
            )
        ]
        x0 = np.array([1.0, 1.0])  # Feasible: 1.0 >= 0.5, y=1.0
        A = np.array([[0.0, 1.0]])  # 0*x + 1*y = 1
        b = np.array([1.0])
    else:
        F = lambda x: x[0] ** 2 + x[1] ** 2
        G = [lambda x: 0.5 - x[0]]
        x0 = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
        A = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)

    res = barrier_method(
        F, G, x0, equality=(A, b), epsilon=1e-5, backend=backend
    )

    ProblemFactory.assert_close(res, [0.5, 1.0], backend, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_strict_feasibility_check_infeasible(backend):
    """
    CRITICAL TEST: The Barrier Method MUST start at a strictly feasible point.
    If we provide an x0 where f_i(x0) > 0, it should raise an error immediately.
    """
    F, G, _ = ProblemFactory.get_1d_linear(backend)

    # Constraint is x >= 1.
    # Try starting at x = 0 (Infeasible)
    if backend == "numpy":
        x0_bad = np.array([0.0])
    else:
        x0_bad = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

    with pytest.raises(ValueError, match="Initial point is not feasible."):
        barrier_method(F, G, x0_bad, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_strict_feasibility_check_boundary(backend):
    """
    CRITICAL TEST: The Barrier Method fails at the boundary because log(0) is undefined.
    If we provide x0 where f_i(x0) == 0, it should raise an error.
    """
    F, G, _ = ProblemFactory.get_1d_linear(backend)

    # Constraint is x >= 1.
    # Try starting exactly at x = 1 (Boundary)
    if backend == "numpy":
        x0_boundary = np.array([1.0])
    else:
        x0_boundary = torch.tensor(
            [1.0], dtype=torch.float64, requires_grad=True
        )

    with pytest.raises(ValueError):
        barrier_method(F, G, x0_boundary, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_return_loss_history(backend):
    """
    Verify return_loss flag returns history of outer loop (centering steps).
    """
    F, G, x0 = ProblemFactory.get_2d_quadratic(backend)

    res, history = barrier_method(F, G, x0, return_loss=True, backend=backend)

    assert isinstance(history, list)
    # Barrier method usually takes a few centering steps (e.g. 5-20)
    assert len(history) > 1
    # Loss should generally decrease (though barrier steps make this complex,
    # the objective f0(x) should trend to optimum)
    assert history[-1] < history[0]


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_barrier_dual_recovery(backend):
    """
    Min x^2 s.t. x >= 1.
    Theoretical Primal: x=1
    Theoretical Dual (lambda): 2.0
    """
    if backend == "numpy":
        F = (
            lambda x: x[0] ** 2,
            lambda x: np.array([2 * x[0]]),
            lambda x: np.array([[2.0]]),
        )
        G = [
            (
                lambda x: 1.0 - x[0],
                lambda x: np.array([-1.0]),
                lambda x: np.array([[0.0]]),
            )
        ]
        x0 = np.array([2.0])
    else:
        F = lambda x: x[0] ** 2
        G = [lambda x: 1.0 - x[0]]
        x0 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

    # Run with dual return enabled
    x_star, (lambdas, mu) = barrier_method(
        F, G, x0, epsilon=1e-5, return_dual=True, backend=backend
    )

    # Check Primal
    ProblemFactory.assert_close(x_star, [1.0], backend, atol=1e-3)

    # Check Dual (Lambda)
    # The lambda list corresponds to constraints in G
    lambda_val = lambdas[0]
    ProblemFactory.assert_close(lambda_val, [2.0], backend, atol=1e-2)
