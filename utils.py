import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np


def _wrap_func(np_func, jnp_func):
    def wrapped(x, *args, **kwargs):
        if isinstance(x, jnp.ndarray):
            return jnp_func(x, *args, **kwargs)
        return np_func(x, *args, **kwargs)

    return wrapped


sum = _wrap_func(np.sum, jnp.sum)
log10 = _wrap_func(np.log10, jnp.log10)
sqrt = _wrap_func(np.sqrt, jnp.sqrt)
