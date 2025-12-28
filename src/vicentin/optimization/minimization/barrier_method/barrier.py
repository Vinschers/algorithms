from typing import Any, Callable, Optional, Sequence

from vicentin.utils import Dispatcher

dispatcher = Dispatcher()


try:
    from barrier_np import barrier_method as barrier_np

    dispatcher.register("numpy", barrier_np)
except ModuleNotFoundError:
    pass

try:
    from barrier_torch import barrier_method as barrier_torch

    dispatcher.register("torch", barrier_torch)
except ModuleNotFoundError:
    pass


def barrier_method(
    F: Sequence[Callable] | Callable,
    G: Sequence[list[Callable]] | Sequence[Callable],
    x0: Any,
    equality: Optional[tuple] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    epsilon: float = 1e-4,
    mu: float = 6,
    return_loss: bool = True,
    backend: Optional[str] = None,
):
    dispatcher.detect_backend(x0, backend)
    x0 = dispatcher.cast_values(x0)

    return dispatcher(
        F, G, x0, equality, max_iter, tol, epsilon, mu, return_loss
    )
