from typing import Any, Callable, Optional, Sequence

from vicentin.utils import Dispatcher


def transform_numpy(*args, **kwargs):
    if args:
        F = args[0]
        G = args[1]

        if not isinstance(F, (list, tuple)) or len(F) < 2:
            raise ValueError("NumPy backend requires F=(f, grad_f)")
        if not isinstance(G, (list, tuple)) or len(G) < 2:
            raise ValueError("Backend requires G=(g, prox_g)")

        f, grad_f = F
        g, prox_g = G

        new_args = (f, grad_f, g, prox_g) + args[2:]
        return new_args, kwargs

    F = kwargs.pop("F")
    G = kwargs.pop("G")
    return args, {
        "f": F[0],
        "grad_f": F[1],
        "g": G[0],
        "prox_g": G[1],
        **kwargs,
    }


def transform_autodiff(*args, **kwargs):
    if args:
        F = args[0]
        G = args[1]

        f = F[0] if isinstance(F, (list, tuple)) else F

        if not isinstance(G, (list, tuple)) or len(G) < 2:
            raise ValueError("Backend requires G=(g, prox_g)")

        g, prox_g = G
        new_args = (f, g, prox_g) + args[2:]
        return new_args, kwargs

    F = kwargs.pop("F")
    G = kwargs.pop("G")
    f = F[0] if isinstance(F, (list, tuple)) else F

    return args, {"f": f, "g": G[0], "prox_g": G[1], **kwargs}


dispatcher = Dispatcher()


try:
    from .proximal_gradient_np import proximal_gradient as proximal_gradient_np

    dispatcher.register("numpy", proximal_gradient_np, transform_numpy)
except ModuleNotFoundError:
    pass

try:
    from .proximal_gradient_torch import (
        proximal_gradient as proximal_gradient_torch,
    )

    dispatcher.register("torch", proximal_gradient_torch, transform_autodiff)
except ModuleNotFoundError:
    pass


def proximal_gradient_descent(
    F: Sequence[Callable] | Callable,
    G: Sequence[Callable],
    x0: Any,
    step_size: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = False,
    backend: Optional[str] = None,
):
    """
    Minimizes a composite function F(x) = f(x) + g(x) using the
    Forward-Backward (Proximal Gradient) algorithm.

    This method is suitable when f(x) is smooth (differentiable) and g(x)
    is non-smooth but has a computable proximal operator (e.g., L1 norm).

    The update step is:
    $x_{k+1} = \\text{prox}_{\\gamma g} (x_k - \\gamma \\nabla f(x_k))$

    Parameters:
    -----------
    F : Sequence[Callable] or Callable
        The smooth part of the objective.
        - NumPy: (f, grad_f)
        - Torch: f
    G : Sequence[Callable]
        The non-smooth part. Must be a tuple `(g, prox_g)`.
        - `g(x)`: Evaluates the function (used for loss tracking).
        - `prox_g(v, gamma)`: Computes $\\text{argmin}_x (g(x) + \\frac{1}{2\\gamma} \\|x - v\\|^2)$.
    x0 : Any
        Initial guess.
    step_size : float, optional
        Fixed step size. If None, uses backtracking line search on f.

    Returns:
    --------
    x : Any
        The approximate minimum.
    loss : List[float], optional
        History of total objective values F(x) + G(x).
    """

    dispatcher.detect_backend(x0, backend)
    x0 = dispatcher.cast_values(x0)

    return dispatcher(F, G, x0, step_size, max_iter, tol, epsilon, return_loss)
