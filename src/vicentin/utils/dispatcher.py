from typing import Any, Callable, Dict, List, Optional, Tuple

SUPPORTED_BACKENDS = {
    "np": "np",
    "numpy": "np",
    "torch": "torch",
    "pytorch": "torch",
    "jax": "jax",
}


class Dispatcher:
    def __init__(self):
        self._registry: Dict[str, Tuple[Callable, Optional[Callable]]] = {}
        self.backend = None

    def register(
        self,
        backend_name: str,
        func: Callable,
        transformer: Optional[Callable] = None,
    ):
        if backend_name.lower() in SUPPORTED_BACKENDS.keys():
            self._registry[SUPPORTED_BACKENDS[backend_name.lower()]] = (
                func,
                transformer,
            )

    def detect_backend(self, arg: Any, backend: Optional[str] = None):
        if backend is not None and backend.lower() in SUPPORTED_BACKENDS.keys():
            self.backend = SUPPORTED_BACKENDS[backend.lower()]
        else:
            self.backend = SUPPORTED_BACKENDS[arg.__class__.__module__.lower()]

    def _cast_value(self, value: Any, target_backend: str) -> Any:
        if value is None:
            return None

        if target_backend == "torch":
            import torch

            if not torch.is_tensor(value):
                return torch.as_tensor(value, dtype=torch.float32)

        elif target_backend == "jax":
            import jax.numpy as jnp

            return jnp.array(value)

        elif target_backend == "np":
            import numpy as np

            if np.ndim(value) == 0:
                return np.float32(value)
            return np.array(value)

        return value

    def cast_values(self, *args) -> List[Any]:
        if self.backend is None:
            raise TypeError(
                "Dispatcher did not detect any backends. Please call `detect_backend` first."
            )

        casted_args = []

        for arg in args:
            casted_args.append(self._cast_value(arg, self.backend))

        if len(args) == 1:
            return casted_args[0]

        return casted_args

    def __call__(self, *args, **kwargs):
        if self.backend is None:
            raise TypeError(
                "Dispatcher did not detect any backends. Please call `detect_backend` first."
            )

        func, transformer = self._registry[self.backend]

        if transformer:
            args, kwargs = transformer(*args, **kwargs)

        return func(*args, **kwargs)
