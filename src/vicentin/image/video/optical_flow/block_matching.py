from vicentin.utils import pad, sum, inf, abs, zeros, median, shape, stack, array
from vicentin.image.utils import img2blocks


def block_matching(ref, cur, block_shape=(8, 8), search_radius=4, cost_method="ssd", lamb=0.0):
    """
    Efficient block-matching motion estimation.

    This function estimates the motion vector field (MVF) between two consecutive frames
    using block-matching. It extracts blocks from the current frame and searches for
    the best match in the reference frame within a given search radius.

    Args:
        ref (ndarray or tf.Tensor): Reference (previous) frame, 2D shape (H, W).
        cur (ndarray or tf.Tensor): Current frame, 2D shape (H, W).
        block_shape (tuple): (bH, bW) size of each block.
        search_radius (int): Radius for block search.
        cost_method (str): "ssd" (Sum of Squared Differences) or "sad" (Sum of Absolute Differences).
        lamb (float): Smoothness weight to penalize large deviations from neighbor median.

    Returns:
        mvf (ndarray or tf.Tensor): The final motion vector field of shape (H, W, 2).
                                    If a block in 'cur' moves by (dy, dx), mvf contains (-dy, -dx).
    """
    H, W = shape(cur)[:2]
    bH, bW = block_shape

    if H % bH != 0 or W % bW != 0:
        raise ValueError("For simplicity, H and W must be multiples of block_shape.")

    nRows = H // bH
    nCols = W // bW
    N = nRows * nCols  # total number of blocks
    lamb_scaled = lamb * bH * bW

    # Pad the reference so we can slice valid areas easily for each displacement
    pad_ref = pad(ref, pad_width=search_radius, mode="edge")

    # ----------------------------------------------------------------
    # 1) Extract Blocks from the Current Frame
    # ----------------------------------------------------------------
    cur_blocks_4d = img2blocks(cur, block_shape, step_row=bH, step_col=bW)
    cur_blocks = cur_blocks_4d.reshape((N, bH, bW))

    # ----------------------------------------------------------------
    # 2) Build a "cost volume" over all candidate displacements
    #    cost_volume shape => (N, nDisp)
    # ----------------------------------------------------------------
    displacements = []
    costs_list = []  # each entry will be shape (N,)

    for drow in range(-search_radius, search_radius + 1):
        for dcol in range(-search_radius, search_radius + 1):
            # Keep track of this candidate displacement
            # Note: final motion is opposite sign => We'll store them as (drow, dcol)
            # but we recall that the user ultimately wants -mvf
            displacements.append((drow, dcol))

            top = search_radius - drow
            left = search_radius - dcol
            ref_shifted = pad_ref[top : top + H, left : left + W]

            # Extract the same blocks from this shifted reference
            ref_blocks_4d = img2blocks(ref_shifted, block_shape, step_row=bH, step_col=bW)
            ref_blocks = ref_blocks_4d.reshape((N, bH, bW))

            if cost_method == "ssd":
                cost_vals = sum((cur_blocks - ref_blocks) ** 2, axis=(1, 2))
            elif cost_method == "sad":
                cost_vals = sum(abs(cur_blocks - ref_blocks), axis=(1, 2))
            else:
                raise ValueError(f"Unrecognized cost method: {cost_method}")

            costs_list.append(cost_vals)

    cost_volume = stack(array(costs_list), axis=-1)  # (N, nDisp)
    nDisp = cost_volume.shape[-1]

    # ----------------------------------------------------------------
    # 3) Compute the final block displacement with neighbor-based penalty
    #    We'll store in 'block_vectors' => (nRows, nCols, 2)
    # ----------------------------------------------------------------
    block_vectors = zeros((nRows, nCols, 2))

    def _penalty(d, pV):
        """Compute smoothness penalty for displacement d vs. median pV."""
        if cost_method == "sad":
            return sum(abs(d - pV)) * lamb_scaled
        else:
            return sum((d - pV) ** 2) * lamb_scaled

    for row in range(nRows):
        for col in range(nCols):
            i = row * nCols + col

            neighbors = []
            if row > 0:
                neighbors.append(block_vectors[row - 1, col])  # top
            if col > 0:
                neighbors.append(block_vectors[row, col - 1])  # left
            if row > 0 and col > 0:
                neighbors.append(block_vectors[row - 1, col - 1])  # top-left

            if neighbors:
                neighbors_tensor = stack(neighbors, axis=0)
                pV = median(neighbors_tensor, axis=0)
            else:
                pV = zeros((2,))

            # (B) Add penalty for each candidate displacement
            # cost_volume[i, :] => shape (nDisp,)
            # We'll add smoothness penalty => shape (nDisp,)
            best_cost = inf
            best_d = (0, 0)

            for disp_idx in range(nDisp):
                drow, dcol = displacements[disp_idx]
                d_vec = array([drow, dcol])  # shape (2,)
                total_cost = cost_volume[i, disp_idx] + _penalty(d_vec, pV)

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_d = (drow, dcol)

            # Store the best displacement
            block_vectors[row, col, 0] = best_d[0]
            block_vectors[row, col, 1] = best_d[1]

    # ----------------------------------------------------------------
    # 4) Expand block_vectors => full-resolution MVF, shape (H, W, 2)
    # ----------------------------------------------------------------
    mvf = zeros((H, W, 2))
    for row in range(nRows):
        for col in range(nCols):
            dy, dx = block_vectors[row, col, 0], block_vectors[row, col, 1]
            # Fill the entire block region with the chosen vector
            r0, c0 = row * bH, col * bW
            mvf[r0 : r0 + bH, c0 : c0 + bW, 0] = dy
            mvf[r0 : r0 + bH, c0 : c0 + bW, 1] = dx

    return -mvf
