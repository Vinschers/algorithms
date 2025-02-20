import numpy as np
import jax.numpy as jnp


def _wrap_func(np_func, jnp_func):
    def wrapped(x, *args, **kwargs):
        if isinstance(x, jnp.ndarray):
            return jnp_func(x, *args, **kwargs)
        return np_func(x, *args, **kwargs)

    return wrapped


array = _wrap_func(np.array, jnp.array)
asarray = _wrap_func(np.asarray, jnp.asarray)
zeros = _wrap_func(np.zeros, jnp.zeros)
arange = _wrap_func(np.arange, jnp.arange)
flip = _wrap_func(np.flip, jnp.flip)
roll = _wrap_func(np.roll, jnp.roll)
argmin = _wrap_func(np.argmin, jnp.argmin)
pad = _wrap_func(np.pad, jnp.pad)
median = _wrap_func(np.median, jnp.median)

unravel_index = _wrap_func(np.unravel_index, jnp.unravel_index)
repeat = _wrap_func(np.repeat, jnp.repeat)
stack = _wrap_func(np.stack, jnp.stack)
tile = _wrap_func(np.tile, jnp.tile)

mean = _wrap_func(np.mean, jnp.mean)
sum = _wrap_func(np.sum, jnp.sum)
log10 = _wrap_func(np.log10, jnp.log10)
sqrt = _wrap_func(np.sqrt, jnp.sqrt)
abs = _wrap_func(np.abs, jnp.abs)

SVD = _wrap_func(np.linalg.svd, jnp.linalg.svd)

inf = np.inf
isnan = _wrap_func(np.isnan, jnp.isnan)
