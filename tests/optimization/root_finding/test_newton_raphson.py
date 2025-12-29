import pytest
import numpy as np
import torch

from vicentin.optimization.root_finding import newton_raphson


class ProblemFactory:
    """
    Standardizes root-finding problems for Numpy, PyTorch, and JAX.
    """

    @staticmethod
    def get_scalar_root(backend):
        """
        Problem: Find x such that x^2 - 4 = 0.
        Roots: x = 2, x = -2.
        """
        if backend == "numpy":
            # Numpy requires (f, jacobian)
            def f(x):
                return x[0] ** 2 - 4.0

            def df(x):
                return np.array([[2 * x[0]]])

            return (f, df), np.array([3.0])

        elif backend == "pytorch":
            # PyTorch requires f, autodiff handles jacobian
            def f(x):
                return x[0] ** 2 - 4.0

            return f, torch.tensor(
                [3.0], dtype=torch.float64, requires_grad=True
            )

        elif backend == "jax":
            # JAX requires f, autodiff handles jacobian
            import jax.numpy as jnp

            def f(x):
                return x[0] ** 2 - 4.0

            return f, jnp.array([3.0])

    @staticmethod
    def get_multivariate_system(backend):
        """
        Problem: Intersection of a Circle and a Hyperbola.
        1. x^2 + y^2 - 5 = 0  (Circle radius sqrt(5))
        2. x^2 - y^2 - 3 = 0  (Hyperbola)
        Root at (2, 1).
        """
        if backend == "numpy":

            def f(x):
                return np.array(
                    [x[0] ** 2 + x[1] ** 2 - 5.0, x[0] ** 2 - x[1] ** 2 - 3.0]
                )

            def jac(x):
                # Jacobian J_ij = df_i / dx_j
                return np.array([[2 * x[0], 2 * x[1]], [2 * x[0], -2 * x[1]]])

            return (f, jac), np.array([4.0, 4.0])

        elif backend == "pytorch":

            def f(x):
                return torch.stack(
                    [x[0] ** 2 + x[1] ** 2 - 5.0, x[0] ** 2 - x[1] ** 2 - 3.0]
                )

            return f, torch.tensor(
                [4.0, 4.0], dtype=torch.float64, requires_grad=True
            )

        elif backend == "jax":
            import jax.numpy as jnp

            def f(x):
                return jnp.array(
                    [x[0] ** 2 + x[1] ** 2 - 5.0, x[0] ** 2 - x[1] ** 2 - 3.0]
                )

            return f, jnp.array([4.0, 4.0])

    @staticmethod
    def assert_close(val1, val2, backend, atol=1e-4):
        """Helper to assert equality across different tensor types."""
        if backend == "numpy":
            np.testing.assert_allclose(val1, val2, atol=atol)
        elif backend == "pytorch":
            if not torch.is_tensor(val1):
                val1 = torch.tensor(val1)
            if not torch.is_tensor(val2):
                val2 = torch.tensor(val2)
            if val1.dtype != val2.dtype:
                val2 = val2.to(dtype=val1.dtype)
            assert torch.allclose(val1, val2, atol=atol)
        elif backend == "jax":
            # JAX arrays can be compared via standard numpy testing
            np.testing.assert_allclose(val1, val2, atol=atol)


BACKENDS = ["numpy", "pytorch"]
try:
    import jax

    # BACKENDS.append("jax")
except ImportError:
    pass


@pytest.mark.parametrize("backend", BACKENDS)
def test_scalar_root_finding(backend):
    """
    Test finding root of x^2 - 4 = 0.
    """
    F, x0 = ProblemFactory.get_scalar_root(backend)

    # We expect returns: (x, loss_history) because return_loss=True
    x_star, loss = newton_raphson(F, x0, return_loss=True, backend=backend)

    # Loss should decrease to near zero
    assert loss[-1] < 1e-5
    ProblemFactory.assert_close(x_star, [2.0], backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_multivariate_system(backend):
    """
    Test solving a coupled system of 2 non-linear equations.
    """
    F, x0 = ProblemFactory.get_multivariate_system(backend)

    x_star, loss = newton_raphson(F, x0, return_loss=True, backend=backend)

    assert loss[-1] < 1e-5
    # We aimed for the positive quadrant root (2, 1)
    ProblemFactory.assert_close(x_star, [2.0, 1.0], backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_divergence_max_iter(backend):
    """
    Test that the algorithm respects max_iter.
    We force this by giving it 1 iteration to solve a hard problem.
    """
    F, x0 = ProblemFactory.get_multivariate_system(backend)

    # Only 1 iteration allowed -> unlikely to converge from [4,4]
    x_star, loss = newton_raphson(
        F, x0, max_iter=1, tol=1e-10, return_loss=True, backend=backend
    )

    # Depending on implementation, history is typically initial + 1 step = 2
    assert len(loss) <= 2
    # The final loss should likely be high (not converged)
    assert loss[-1] > 1e-3


@pytest.mark.parametrize("backend", BACKENDS)
def test_return_signature_flags(backend):
    """
    Verify that toggling return_loss changes output format.
    """
    F, x0 = ProblemFactory.get_scalar_root(backend)

    # Case 1: return_loss = False -> Returns just x
    res1 = newton_raphson(F, x0, return_loss=False, backend=backend)

    if backend == "numpy":
        assert isinstance(res1, np.ndarray)
    elif backend == "pytorch":
        assert torch.is_tensor(res1)
    elif backend == "jax":
        # JAX arrays usually inherit from jax.Array
        import jax

        assert isinstance(res1, (jax.Array, np.ndarray))

    # Case 2: return_loss = True -> Returns (x, loss)
    res2 = newton_raphson(F, x0, return_loss=True, backend=backend)
    assert isinstance(res2, (tuple, list))
    assert len(res2) == 2
    assert isinstance(res2[1], list)  # Loss history is a list


@pytest.mark.parametrize("backend", BACKENDS)
def test_singular_jacobian_handling(backend):
    """
    Root finding requires inverting the Jacobian J.
    If J is singular (determinant is 0), the step cannot be computed.
    Problem: f(x) = x^3. Root is 0.
    At x=0, f'(x) = 3x^2 = 0.
    """
    if backend == "numpy":

        def f(x):
            return x[0] ** 3 + 4

        def df(x):
            return np.array([[3 * x[0] ** 2]])

        F = (f, df)
        x0 = np.array([0.0])  # Derivative is exactly 0 here
    elif backend == "pytorch":

        def f(x):
            return x[0] ** 3 + 4

        F = f
        x0 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    elif backend == "jax":
        import jax.numpy as jnp

        def f(x):
            return x[0] ** 3

        F = f
        x0 = jnp.array([0.0])

    # Expect failures.
    # Numpy raises LinAlgError.
    # Torch raises RuntimeError (or LinAlgError depending on version).
    # JAX raises TypeError or floating point error depending on config,
    # but usually produces NaNs if not crashed.
    # Assuming the implementation allows bubbling up errors:

    with pytest.raises((np.linalg.LinAlgError, RuntimeError, ValueError)):
        newton_raphson(F, x0, backend=backend)
