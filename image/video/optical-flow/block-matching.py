import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np

from image.utils import img2blocks


def _block_matching_numpy(img1, img2, block_shape=(8, 8), search_radius=16):
    """
    Block Matching Motion Estimation (NumPy Version).

    Args:
        img1 (np.ndarray): First frame (reference).
        img2 (np.ndarray): Second frame (current).
        block_shape (tuple): Block size (height, width).
        search_radius (int): Search radius for block matching.

    Returns:
        np.ndarray: Motion Vector Field.
    """
    img1, img2 = np.asarray(img1), np.asarray(img2)
    H, W = img2.shape
    bH, bW = block_shape

    blocks_second = np.asarray(img2blocks(img2, block_shape))

    padded_img1 = np.pad(img1, ((search_radius, search_radius), (search_radius, search_radius)), mode="edge")

    blocks_first = np.lib.stride_tricks.as_strided(
        padded_img1,
        shape=(H - bH + 1, W - bW + 1, bH, bW),
        strides=(padded_img1.strides[0], padded_img1.strides[1], padded_img1.strides[0], padded_img1.strides[1]),
    )

    search_grid_x = np.arange(-search_radius, search_radius + 1)
    search_grid_y = np.arange(-search_radius, search_radius + 1)

    costs = np.array(
        [
            np.sum((blocks_second - blocks_first[i : i + blocks_second.shape[0], j : j + blocks_second.shape[1]]) ** 2, axis=(2, 3))
            for i in search_grid_x
            for j in search_grid_y
        ]
    )

    min_indices = np.argmin(costs, axis=0)
    dx, dy = np.unravel_index(min_indices, (len(search_grid_x), len(search_grid_y)))
    dx, dy = search_grid_x[dx], search_grid_y[dy]

    mvf = np.zeros((H, W, 2))
    row_indices, col_indices = np.arange(0, H, bH)[:, None], np.arange(0, W, bW)

    mvf[row_indices : row_indices + bH, col_indices : col_indices + bW, 0] = -dx[:, :, None, None]
    mvf[row_indices : row_indices + bH, col_indices : col_indices + bW, 1] = -dy[:, :, None, None]

    return mvf


def block_matching(img1, img2, block_shape=(8, 8), search_radius=16):
    """
    Efficient block-matching motion estimation.

    Args:
        img1 (np.ndarray or jnp.ndarray): First frame (reference).
        img2 (np.ndarray or jnp.ndarray): Second frame (current).
        block_shape (tuple): Block size (height, width).
        search_radius (int): Search radius for block matching.

    Returns:
        mvf (np.ndarray or jnp.ndarray): Motion Vector Field.
    """

    if isinstance(img1, jnp.ndarray) and isinstance(img2, jnp.ndarray):
        pass

    return _block_matching_numpy(img1, img2, block_shape, search_radius)
