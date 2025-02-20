from vicentin.utils import pad, arange, sum, argmin, unravel_index, repeat, inf, abs, zeros, array, median
from vicentin.image.utils import img2blocks


def _block_matching_no_reg(ref, cur, block_shape=(8, 8), search_radius=4, cost_method="ssd"):
    """
    Efficient block-matching motion estimation without regularization.

    This function estimates the motion vector field (MVF) between two consecutive frames
    using block-matching. It extracts blocks from the current frame and searches for
    the best match in the reference frame within a given search radius.

    Args:
        ref (np.ndarray or jnp.ndarray): The reference (previous) frame, a 2D array.
        cur (np.ndarray or jnp.ndarray): The current frame, a 2D array.
        block_shape (tuple): The size of the blocks (block_height, block_width).
        search_radius (int): The radius of the search window for block matching.
        cost_method (str): The cost function to evaluate block similarity.
                          Options:
                          - "ssd" (Sum of Squared Differences)
                          - "sad" (Sum of Absolute Differences)

    Returns:
        mvf (np.ndarray): Motion Vector Field.
    """
    H, W = cur.shape[:2]
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

    if cost_method == "ssd":
        cost = sum(
            (candidate_blocks - cur_blocks[:, :, None, None, :, :]) ** 2, axis=(-2, -1)
        )  # shape: (n_rows, n_cols, 2*search_radius+1, 2*search_radius+1)
    elif cost_method == "sad":
        cost = sum(
            abs(candidate_blocks - cur_blocks[:, :, None, None, :, :]), axis=(-2, -1)
        )  # shape: (n_rows, n_cols, 2*search_radius+1, 2*search_radius+1)
    else:
        raise ValueError(f"Unrecognized cost method: {cost_method}.")

    best_idx = argmin(cost.reshape(n_rows, n_cols, -1), axis=2)

    D = 2 * search_radius + 1
    dy, dx = unravel_index(best_idx, (D, D))  #  dx and dy have shape (n_rows, n_cols)

    # Expand dy and dx to full image size
    dy = repeat(repeat(dy - search_radius, bH, axis=0), bW, axis=1)
    dx = repeat(repeat(dx - search_radius, bH, axis=0), bW, axis=1)

    mvf = zeros((H, W, 2))
    mvf[: n_rows * bH, : n_cols * bW, 0] = dy
    mvf[: n_rows * bH, : n_cols * bW, 1] = dx

    return mvf


def block_matching(ref, cur, block_shape=(8, 8), search_radius=4, cost_method="ssd", lamb=0.0):
    """
    Efficient block-matching motion estimation.

    This function estimates the motion vector field (MVF) between two consecutive frames
    using block-matching. It extracts blocks from the current frame and searches for
    the best match in the reference frame within a given search radius.

    Args:
        ref (np.ndarray or jnp.ndarray): The reference (previous) frame, a 2D array.
        cur (np.ndarray or jnp.ndarray): The current frame, a 2D array.
        block_shape (tuple): The size of the blocks (block_height, block_width).
        search_radius (int): The radius of the search window for block matching.
        cost_method (str): The cost function to evaluate block similarity.
                          Options:
                          - "ssd" (Sum of Squared Differences)
                          - "sad" (Sum of Absolute Differences)

    Returns:
        mvf (np.ndarray): Motion Vector Field.
    """

    if lamb == 0.0:
        return _block_matching_no_reg(ref, cur, block_shape, search_radius, cost_method)

    H, W = cur.shape[:2]
    bH, bW = block_shape

    lamb *= bH * bW

    pad_ref = pad(ref, pad_width=search_radius, mode="edge")
    mvf = zeros((H, W, 2))

    for r in range(0, H, bH):
        for c in range(0, W, bW):
            # current block selection
            B = cur[r : r + bH, c : c + bW]

            neighbors = []
            if r >= bH:
                neighbors.append(- mvf[r - bH, c])  # Top
            if c >= bW:
                neighbors.append(- mvf[r, c - bW])  # Left
            if r >= bH and c >= bW:
                neighbors.append(- mvf[r - bH, c - bW])  # Top-left

            # Compute predictor motion vector (pV) using the median of neighbors
            pV = median(neighbors, axis=0) if neighbors else zeros(2)

            min_cost = inf
            d = [0, 0]

            # Loop on candidate vectors
            for drow in range(-search_radius, search_radius + 1):
                for dcol in range(-search_radius, search_radius + 1):
                    p, q = search_radius + r + drow, search_radius + c + dcol
                    R = pad_ref[p : p + bH, q : q + bW]

                    if cost_method == "ssd":
                        cost = sum((B - R) ** 2) + lamb * sum((d - pV) ** 2)
                    elif cost_method == "sad":
                        cost = sum(abs(B - R)) + lamb * sum(abs(d - pV))
                    else:
                        raise ValueError(f"Unrecognized cost method: {cost_method}.")

                    if cost < min_cost:
                        d = [drow, dcol]
                        min_cost = cost

            mvf[r : r + bH, c : c + bW, 0] = d[0]
            mvf[r : r + bH, c : c + bW, 1] = d[1]

    return mvf
