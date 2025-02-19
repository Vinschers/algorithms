import numpy as np
from scipy.signal import convolve2d

try:
    import jax.numpy as jnp
    from jax.scipy.signal import convolve2d as jax_convolve2d
except ImportError:
    jnp = np
    jax_convolve2d = convolve2d

from vicentin.utils import sum, log10, sqrt


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
    new_strides = (img.strides[0] * step_row, img.strides[1] * step_col, img.strides[0], img.strides[1])

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

    blocks = jnp.array(
        [
            [img[i : i + block_height, j : j + block_width] for j in range(0, W - block_width + 1, step_col)]
            for i in range(0, H - block_height + 1, step_row)
        ]
    )
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


def convolve(img, kernel, mode="same"):
    """
    Convolve an image with a kernel using 2D convolution.

    This function supports both NumPy and JAX arrays. If both the image and kernel
    are JAX arrays, it uses `jax_convolve2d`; otherwise, it falls back to SciPy's
    `convolve2d`.

    Parameters
    ----------
    img : numpy.ndarray or jax.numpy.ndarray
        The input image.
    kernel : numpy.ndarray or jax.numpy.ndarray
        The convolution kernel.
    mode : str, optional
        The mode of convolution (e.g., "same", "valid"). Default is "same".

    Returns
    -------
    numpy.ndarray or jax.numpy.ndarray
        The convolved image.
    """
    if isinstance(img, jnp.ndarray):
        return jax_convolve2d(img, kernel, mode)  # type: ignore

    return convolve2d(img, kernel, mode)


def PSNR(img1, img2):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is defined as:
        PSNR = 20 * log10(max_I / sqrt(MSE))
    where MSE (Mean Squared Error) is computed between img1 and img2, and max_I is the
    maximum possible pixel value (assumed here as the maximum value in img1).

    Parameters
    ----------
    img1 : numpy.ndarray or jax.numpy.ndarray
        The first image.
    img2 : numpy.ndarray or jax.numpy.ndarray
        The second image. Must have the same shape as img1.

    Returns
    -------
    float
        The PSNR value in decibels (dB).

    Raises
    ------
    AssertionError
        If the input images do not have the same shape.
    """
    assert img1.shape == img2.shape, "Images must have the same dimensions."
    H, W = img1.shape[:2]

    mse = sum((img1 - img2) ** 2) / (H * W)
    max_I = img1.max()
    return 20 * log10(max_I / sqrt(mse))
