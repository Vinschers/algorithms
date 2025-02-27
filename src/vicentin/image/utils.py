import numpy as np
import tensorflow as tf
from skimage.transform import resize as skimage_resize
from skimage.color import rgb2gray
from scipy.ndimage import convolve as scipy_convolve, gaussian_filter as scipy_gaussian_filter

from vicentin.utils import _wrap_func, shape, expand_dims, copy, asarray, meshgrid, exp, arange, sum


resize = _wrap_func(skimage_resize, tf.image.resize)
grayscale = _wrap_func(rgb2gray, tf.image.rgb_to_grayscale)


def flip(image, horizontal=True):
    """
    Flips an image horizontally or vertically.

    This function flips an image while maintaining backend compatibility.

    Args:
        image (ndarray | Tensor): A 2D or 3D image array.
        horizontal (bool, optional): If True, performs a horizontal flip. If False, performs a vertical flip. Defaults to True.

    Returns:
        ndarray | Tensor: The flipped image.
    """
    return _wrap_func(
        lambda img: np.flip(img, axis=1) if horizontal else np.flip(img, axis=0),
        lambda img: tf.image.flip_left_right(img) if horizontal else tf.image.flip_up_down(img),
    )(image)


def convolve(image, kernel, padding="same", strides=1, **kwargs):
    """
    Applies a 2D convolution to an image using a given kernel.

    This function performs convolution while ensuring backend compatibility.

    Args:
        image (ndarray | Tensor): A 2D or 3D image array (grayscale or multi-channel).
        kernel (ndarray | Tensor): A 2D array representing the convolution filter.
        padding (str, optional): Padding mode for convolution. Defaults to "same".
        strides (int, optional): The step size for moving the kernel across the image. Defaults to 1.
        **kwargs: Additional arguments passed to `scipy.ndimage.convolve` (NumPy) or `tf.nn.conv2d` (TensorFlow).

    Returns:
        ndarray | Tensor: The convolved image.
    """

    def _convolve_np(img, k):
        """NumPy-based convolution using SciPy's ndimage.convolve (vectorized)."""
        return scipy_convolve(img, k, mode=padding, **kwargs)

    def _convolve_tf(img, k):
        """TensorFlow-based convolution using tf.nn.conv2d."""
        img = expand_dims(img, axis=0)  # Add batch dimension if missing
        if img.ndim == 3:  # If grayscale, add channel dimension
            img = expand_dims(img, axis=-1)
        k = expand_dims(expand_dims(k, axis=-1), axis=-1)  # Convert kernel to 4D (TensorFlow format)
        convolved = tf.nn.conv2d(img, k, strides=[1, strides, strides, 1], padding=padding.upper(), **kwargs)
        return convolved[0]  # Remove batch dimension

    return _wrap_func(_convolve_np, _convolve_tf)(image, kernel)


def gaussian_filter(image, sigma):
    """
    Applies a Gaussian blur to an image using a given standard deviation (sigma).

    This function performs Gaussian filtering while ensuring backend compatibility.

    Args:
        image (ndarray | Tensor): A 2D or 3D image array (grayscale or multi-channel).
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        ndarray | Tensor: The blurred image.
    """

    def _gaussian_kernel(size, sigma):
        """Generates a 2D Gaussian kernel using backend-agnostic operations."""
        x = arange(-size // 2 + 1, size // 2 + 1, default_fallback="tf")
        x, y = meshgrid(x, x)
        kernel = exp(-(x**2 + y**2) / (2.0 * sigma**2))
        return kernel / sum(kernel)

    def _gaussian_np(img, s):
        """NumPy-based Gaussian filtering using SciPy."""
        return scipy_gaussian_filter(img, sigma=s)

    def _gaussian_tf(img, s):
        """TensorFlow-based Gaussian filtering using convolution."""
        size = int(6 * s) + 1  # Kernel size approximation
        kernel = _gaussian_kernel(size, s)
        return convolve(img, kernel)

    return _wrap_func(_gaussian_np, _gaussian_tf)(image, sigma)


def img2blocks(img, block_shape, step_row, step_col):
    """
    Extracts non-overlapping or overlapping blocks from an image.

    This function extracts image patches while maintaining backend compatibility.

    Args:
        img (ndarray | Tensor): Input image.
        block_shape (tuple): Block size (height, width).
        step_row (int): Step size in row direction.
        step_col (int): Step size in column direction.

    Returns:
        ndarray | Tensor: Extracted blocks.
    """

    def _img2blocks_np(img, block_shape, step_row, step_col):
        """NumPy implementation using stride tricks."""
        img = asarray(img)
        H, W = shape(img)[:2]
        bH, bW = block_shape
        n_rows = (H - bH) // step_row + 1
        n_cols = (W - bW) // step_col + 1
        new_shape = (n_rows, n_cols, bH, bW)
        new_strides = (
            img.strides[0] * step_row,
            img.strides[1] * step_col,
            img.strides[0],
            img.strides[1],
        )
        blocks = np.lib.stride_tricks.as_strided(img, shape=new_shape, strides=new_strides, writeable=False)
        return copy(blocks)

    def _img2blocks_tf(img, block_shape, step_row, step_col):
        """TensorFlow implementation using tf.image.extract_patches."""
        img = expand_dims(img, axis=0)
        img = expand_dims(img, axis=-1) if len(shape(img)) == 3 else img
        bH, bW = block_shape
        blocks = tf.image.extract_patches(images=img, sizes=[1, bH, bW, 1], strides=[1, step_row, step_col, 1], rates=[1, 1, 1, 1], padding="VALID")
        return blocks[0]

    return _wrap_func(_img2blocks_np, _img2blocks_tf)(img, block_shape, step_row, step_col)


def get_neighbors(image, row, col, depth=None, neighborhood=4):
    """
    Retrieves the neighboring pixels of a given pixel in a 2D or 3D image.

    Args:
        image (ndarray | Tensor): The input image (2D or 3D).
        row (int): Row index.
        col (int): Column index.
        depth (int, optional): Depth index (for 3D images). Defaults to None.
        neighborhood (int, optional): Neighborhood type (4 or 8). Defaults to 4.

    Returns:
        list[tuple[int, int, int]]: List of valid neighboring coordinates.
    """

    H, W = shape(image)[:2]
    L = shape(image)[2] if len(shape(image)) == 3 else 1
    k = depth if depth is not None else 0
    moves = (
        np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
        if neighborhood == 4
        else np.array([[i, j, m] for i in [-1, 0, 1] for j in [-1, 0, 1] for m in [-1, 0, 1] if not (i == j == m == 0)])
    )
    neighbors = np.array([row, col, k]) + moves
    return [
        tuple(n)
        for n in neighbors[
            (0 <= neighbors[:, 0])
            & (neighbors[:, 0] < H)
            & (0 <= neighbors[:, 1])
            & (neighbors[:, 1] < W)
            & (0 <= neighbors[:, 2])
            & (neighbors[:, 2] < L)
        ]
    ]
