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
sign = _wrap_func(np.sign, jnp.sign)
maximum = _wrap_func(np.maximum, jnp.maximum)
where = _wrap_func(np.where, jnp.where)

unravel_index = _wrap_func(np.unravel_index, jnp.unravel_index)
repeat = _wrap_func(np.repeat, jnp.repeat)
stack = _wrap_func(np.stack, jnp.stack)
tile = _wrap_func(np.tile, jnp.tile)

mean = _wrap_func(np.mean, jnp.mean)
sum = _wrap_func(np.sum, jnp.sum)
log10 = _wrap_func(np.log10, jnp.log10)
sqrt = _wrap_func(np.sqrt, jnp.sqrt)
abs = _wrap_func(np.abs, jnp.abs)
norm = _wrap_func(np.linalg.norm, jnp.linalg.norm)
dot = _wrap_func(np.dot, jnp.dot)

SVD = _wrap_func(np.linalg.svd, jnp.linalg.svd)

inf = np.inf
isnan = _wrap_func(np.isnan, jnp.isnan)


def soft_threshold(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
