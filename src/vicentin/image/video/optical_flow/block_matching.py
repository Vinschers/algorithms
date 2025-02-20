from vicentin.utils import pad, arange, sum, argmin, unravel_index, tile, stack
from vicentin.image.utils import img2blocks


def block_matching(ref, cur, block_shape=(8, 8), search_radius=4):
    """
    Efficient block-matching motion estimation.

    Args:
        ref (np.ndarray or jnp.ndarray): First (reference) frame, 2D array.
        cur (np.ndarray or jnp.ndarray): Second (current) frame, 2D array.
        block_shape (tuple): (block_height, block_width).
        search_radius (int): Search radius for block matching.

    Returns:
        mvf (np.ndarray): Motion Vector Field.
    """
    bH, bW = block_shape

    pad_ref = pad(ref, pad_width=search_radius, mode="edge")

    cur_blocks = img2blocks(cur, block_shape)
    ref_blocks = img2blocks(pad_ref, block_shape, 1, 1)

    n_rows, n_cols = cur_blocks.shape[:2]

    row_idx = bH * arange(n_rows) + search_radius
    col_idx = bW * arange(n_cols) + search_radius
    search_range = arange(-search_radius, search_radius + 1)

    candidate_blocks = ref_blocks[
        row_idx[:, None, None, None] + search_range[None, None, :, None],  # shape: (n_rows, 1, s, 1)
        col_idx[None, :, None, None] + search_range[None, None, None, :],  # shape: (1, n_cols, 1, s)
    ]  # shape: (n_rows, n_cols, 2*search_radius+1, 2*search_radius+1, bH, bW)

    cost = sum((candidate_blocks - cur_blocks[:, :, None, None, :, :]) ** 2, axis=(-2, -1))  # shape: (n_rows, n_cols, 2*search_radius+1, 2*search_radius+1)

    best_idx = argmin(cost.reshape(n_rows, n_cols, -1), axis=2)

    D = 2 * search_radius + 1
    dy, dx = unravel_index(best_idx, (D, D)) #  dx and dy have shape (n_rows, n_cols)

    # Expand dy and dx to be (H, W)
    dy = (dy - search_radius)[:, :, None, None]
    dx = (dx - search_radius)[:, :, None, None]
    dy = tile(dy, (1, 1, bH, bW))
    dx = tile(dx, (1, 1, bH, bW))

    mvf = stack([dy, dx], axis=-1)

    return mvf
