import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def _wrap_func(np_func, tf_func):
    """
    Wraps NumPy and TensorFlow functions into a single function that selects the appropriate backend.

    Args:
        np_func: The NumPy function.
        tf_func: The TensorFlow function.

    Returns:
        A function that automatically detects the backend based on the input type and scalar flag.
    """

    def wrapped(x, *args, default_for_scalars="np", **kwargs):
        if isinstance(x, tf.Tensor):
            return tf_func(x, *args, **kwargs)
        elif isinstance(x, np.ndarray):
            return np_func(x, *args, **kwargs)
        elif isinstance(x, (int, float)):
            if default_for_scalars == "np":
                return np_func(x, *args, **kwargs)  # Use NumPy for scalars
            elif default_for_scalars == "tf":
                return tf_func(x, *args, **kwargs)  # Use TensorFlow for scalars
            else:
                raise ValueError("default_for_scalars must be 'np' or 'tf'")
        else:
            raise TypeError(f"Unsupported type: {type(x)}. Expected numpy.ndarray, tf.Tensor, or scalar.")

    return wrapped


# --- Array Operations ---
arange = _wrap_func(np.arange, tf.range)
array = _wrap_func(np.array, tf.convert_to_tensor)
asarray = _wrap_func(np.asarray, tf.convert_to_tensor)
expand_dims = _wrap_func(np.expand_dims, tf.expand_dims)
pad = _wrap_func(np.pad, tf.pad)
repeat = _wrap_func(np.repeat, tf.repeat)
stack = _wrap_func(np.stack, tf.stack)
tile = _wrap_func(np.tile, tf.tile)
zeros = _wrap_func(np.zeros, tf.zeros)
zeros_like = _wrap_func(np.zeros_like, tf.zeros_like)
ones = _wrap_func(np.ones, tf.ones)


# --- Indexing and Sorting ---
def _argsort_np(x, direction="ASCENDING", *args, **kwargs):
    if direction == "DESCENDING":
        return np.argsort(x, *args, **kwargs)[::-1]
    return np.argsort(x, *args, **kwargs)


def _argsort_tf(x, direction="ASCENDING", *args, **kwargs):
    return tf.argsort(x, *args, direction=direction, **kwargs)


argsort = _wrap_func(_argsort_np, _argsort_tf)

index_select = _wrap_func(lambda arr, idx, axis: arr.take(idx, axis=axis), lambda arr, idx, axis: tf.gather(arr, idx, axis=axis))
where = _wrap_func(np.where, tf.where)

# --- Linear Algebra ---
dot = _wrap_func(np.dot, tf.linalg.matmul)
eigh = _wrap_func(np.linalg.eigh, tf.linalg.eigh)
inv = _wrap_func(np.linalg.inv, tf.linalg.inv)
matmul = _wrap_func(np.matmul, tf.linalg.matmul)
norm = _wrap_func(np.linalg.norm, tf.norm)
solve = _wrap_func(np.linalg.solve, tf.linalg.solve)
SVD = _wrap_func(np.linalg.svd, tf.linalg.svd)

# --- Mathematical Operations ---
abs = _wrap_func(np.abs, tf.abs)
exp = _wrap_func(np.exp, tf.exp)
flip = _wrap_func(np.flip, tf.reverse)
log10 = _wrap_func(np.log10, lambda x: tf.math.log(x) / tf.math.log(10.0))
max = _wrap_func(np.max, tf.reduce_max)
maximum = _wrap_func(np.maximum, tf.maximum)
mean = _wrap_func(np.mean, tf.reduce_mean)
median = _wrap_func(np.median, lambda x, axis=None, keepdims=False: tfp.stats.percentile(x, 50.0, axis=axis, keepdims=keepdims))
min = _wrap_func(np.min, tf.reduce_min)
prod = _wrap_func(np.prod, tf.reduce_prod)
sign = _wrap_func(np.sign, tf.sign)
sqrt = _wrap_func(np.sqrt, tf.sqrt)
sum = _wrap_func(np.sum, tf.reduce_sum)

# --- Statistical Functions ---
argmin = _wrap_func(np.argmin, tf.argmin)
isnan = _wrap_func(np.isnan, tf.math.is_nan)

# --- Utility Functions ---
copy = _wrap_func(lambda x: x.copy(), tf.identity)
scalar = _wrap_func(lambda x: float(x), lambda x: x.numpy().item())
shape = _wrap_func(np.shape, lambda x: x.shape.as_list())
cast = _wrap_func(np.asarray, tf.cast)


def _roll_tf(x, shift, axis):
    """TensorFlow-based roll function using tf.concat and slicing."""
    shape_x = shape(x)  # Use our shape function for backend compatibility

    # Normalize the shift to handle both positive and negative shifts
    shift = shift % shape_x[axis]

    # Slicing and concatenating along the specified axis
    left = tf.strided_slice(x, [0] * len(shape_x), shape_x)  # Slice from the start to shift
    right = tf.strided_slice(x, [shift] * len(shape_x), shape_x)  # Slice from shift to end

    # Concatenate the two parts along the specified axis
    return tf.concat([right, left], axis=axis)


roll = _wrap_func(np.roll, _roll_tf)


# --- Coordinate Grid Operations ---
meshgrid = _wrap_func(np.meshgrid, tf.meshgrid)

# --- Constants ---
inf = np.inf


# --- Custom Operations ---
def soft_threshold(x, alpha):
    return sign(x) * maximum(abs(x) - alpha, 0)
