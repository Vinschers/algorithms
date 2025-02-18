import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    jnp = np


def _img2blocks_numpy(img, block_shape, step_row, step_col):
    """
    Extracts non-overlapping or overlapping blocks from an image using NumPy (CPU).

    Args:
        img (np.ndarray): Input image.
        block_shape (tuple): Block size (height, width).
        step_row (int): Step size in row direction.
        step_col (int): Step size in column direction.

    Returns:
        np.ndarray: Extracted blocks.
    """
    img = np.asarray(img)
    H, W = img.shape
    block_height, block_width = block_shape

    # Compute number of blocks
    n_rows = (H - block_height) // step_row + 1
    n_cols = (W - block_width) // step_col + 1

    # Use as_strided() for efficient block extraction
    new_shape = (n_rows, n_cols, block_height, block_width)
    new_strides = (img.strides[0] * step_row,
                   img.strides[1] * step_col,
                   img.strides[0],
                   img.strides[1])

    return np.lib.stride_tricks.as_strided(img, shape=new_shape, strides=new_strides, writeable=False)


def _img2blocks_jax(img, block_shape, step_row, step_col):
    """
    Extracts non-overlapping or overlapping blocks from an image using JAX (GPU/TPU).

    Args:
        img (jnp.ndarray): Input image.
        block_shape (tuple): Block size (height, width).
        step_row (int): Step size in row direction.
        step_col (int): Step size in column direction.

    Returns:
        jnp.ndarray: Extracted blocks.
    """
    img = jnp.asarray(img)
    H, W = img.shape
    block_height, block_width = block_shape

    # Manually extract blocks using slicing
    blocks = jnp.array([
        [img[i:i + block_height, j:j + block_width] for j in range(0, W - block_width + 1, step_col)]
        for i in range(0, H - block_height + 1, step_row)
    ])
    return blocks


def img2blocks(img, block_shape, step_row=-1, step_col=-1):
    """
    Extracts non-overlapping or overlapping blocks from an image.

    Args:
        img (np.ndarray or jnp.ndarray): Input image.
        block_shape (tuple): Block size (height, width).
        step_row (int, optional): Step size in row direction. Defaults to block height.
        step_col (int, optional): Step size in column direction. Defaults to block width.
        backend (str, optional): "cpu" (NumPy) or "jax" (JAX). If None, auto-detect.

    Returns:
        np.ndarray or jnp.ndarray: Extracted blocks.
    """

    if step_row == -1:
        step_row = block_shape[0]
    if step_col == -1:
        step_col = block_shape[1]

    if isinstance(img, jnp.ndarray):
        return _img2blocks_jax(img, block_shape, step_row, step_col)

    return _img2blocks_numpy(img, block_shape, step_row, step_col)
