from vicentin.utils import pad, sum, inf, abs, zeros, median


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
                neighbors.append(mvf[r - bH, c])  # Top
            if c >= bW:
                neighbors.append(mvf[r, c - bW])  # Left
            if r >= bH and c >= bW:
                neighbors.append(mvf[r - bH, c - bW])  # Top-left

            # Compute predictor motion vector (pV) using the median of neighbors
            pV = median(neighbors, axis=0) if neighbors else zeros(2)

            min_cost = inf
            best_d = [0, 0]

            # Loop on candidate vectors
            for drow in range(-search_radius, search_radius + 1):
                for dcol in range(-search_radius, search_radius + 1):
                    p, q = search_radius + r + drow, search_radius + c + dcol
                    d = [-drow, -dcol]

                    R = pad_ref[p : p + bH, q : q + bW]

                    if cost_method == "ssd":
                        cost = sum((B - R) ** 2) + lamb * sum((d - pV) ** 2)
                    elif cost_method == "sad":
                        cost = sum(abs(B - R)) + lamb * sum(abs(d - pV))
                    else:
                        raise ValueError(f"Unrecognized cost method: {cost_method}.")

                    if cost < min_cost:
                        best_d = d
                        min_cost = cost

            mvf[r : r + bH, c : c + bW, 0] = best_d[0]
            mvf[r : r + bH, c : c + bW, 1] = best_d[1]

    return -mvf
