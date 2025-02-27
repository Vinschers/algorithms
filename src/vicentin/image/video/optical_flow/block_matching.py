from vicentin.utils import pad, sum, abs, zeros, median, shape, stack, array, cast, reshape, argmin, repeat
from vicentin.image.utils import img2blocks


def _check_and_prepare_shapes(cur, block_shape):
    """
    Checks and returns dimensions for block matching.

    Returns:
        (H, W, bH, bW, nRows, nCols, N)
    """
    H, W = shape(cur)[:2]
    bH, bW = block_shape

    if H % bH != 0 or W % bW != 0:
        raise ValueError("For simplicity, H and W must be multiples of block_shape.")

    nRows = H // bH
    nCols = W // bW
    N = nRows * nCols  # total number of blocks
    return H, W, bH, bW, nRows, nCols, N


def _pad_reference(ref, search_radius):
    """
    Pads the reference frame so that searching near edges is possible.
    """
    pad_width = [[search_radius, search_radius], [search_radius, search_radius]]
    return pad(ref, pad_width, mode="edge")


def _extract_blocks(frame, block_shape, step_row, step_col):
    """
    Extracts 2D blocks from 'frame' (H,W) => returns a 4D structure.
    The caller may reshape further into (N, bH, bW).
    """
    blocks_4d = img2blocks(frame, block_shape, step_row=step_row, step_col=step_col)
    return blocks_4d


def _build_cost_volume(pad_ref, cur_blocks, H, W, block_shape, search_radius, cost_method, dtype_for_disp):
    """
    Builds the cost volume of shape (N, nDisp) plus the array of candidate displacements.

    Args:
        pad_ref: Padded reference frame (2D array/tensor).
        cur_blocks: (N, bH, bW) blocks from the 'cur' frame.
        H, W: height/width of the unpadded reference.
        block_shape: (bH, bW).
        search_radius: +/- offset for block search.
        cost_method: "ssd" or "sad".
        dtype_for_disp: dtype to cast displacements (should match 'cur' dtype).

    Returns:
        cost_volume (N, nDisp)
        all_disp (nDisp, 2)   # each displacement (drow, dcol)
    """
    N = shape(cur_blocks)[0]  # total number of blocks
    bH, bW = block_shape

    displacements = []
    costs_list = []

    # For each candidate displacement, shift the padded ref, extract blocks, compute cost
    for drow in range(-search_radius, search_radius + 1):
        for dcol in range(-search_radius, search_radius + 1):
            # Store displacements as e.g. float32 if your 'cur' is float32
            displacements.append(cast((drow, dcol), dtype_for_disp))

            # Slice the padded reference to align with (drow, dcol)
            top = search_radius - drow
            left = search_radius - dcol
            ref_shifted = pad_ref[top : top + H, left : left + W]

            # Extract blocks from shifted ref => shape (N, bH, bW)
            ref_blocks_4d = _extract_blocks(ref_shifted, block_shape, step_row=bH, step_col=bW)
            ref_blocks = reshape(ref_blocks_4d, (N, bH, bW))

            # Compute cost
            if cost_method == "ssd":
                cost_vals = sum((cur_blocks - ref_blocks) ** 2, axis=(1, 2))
            elif cost_method == "sad":
                cost_vals = sum(abs(cur_blocks - ref_blocks), axis=(1, 2))
            else:
                raise ValueError(f"Unrecognized cost method: {cost_method}")

            costs_list.append(cost_vals)

    # Stack => (N, nDisp)
    cost_volume = stack(array(costs_list), axis=-1)
    # Turn displacements into a (nDisp, 2) array/tensor
    all_disp = array(displacements)

    return cost_volume, all_disp


def _compute_blockwise_disp(cost_volume, all_disp, nRows, nCols, cost_method, lamb_scaled):
    """
    Computes blockwise displacements with neighbor smoothing, in a single pass from top-left to bottom-right.

    Args:
        cost_volume: shape (N, nDisp)
        all_disp:    shape (nDisp, 2)
        nRows, nCols (int): grid dimensions (blocks)
        cost_method (str): "ssd" or "sad"
        lamb_scaled (float): lambda * block_area

    Returns:
        block_vectors: shape (nRows, nCols, 2)
    """
    # Number of blocks = nRows * nCols
    N = nRows * nCols

    # We'll store each block's (dy, dx) in a Python list of length N.
    # This is more efficient than a nested list [ [ ... ] for row in ... ]
    # and also avoids issues with TF's immutability if we tried to update a tensor directly.
    block_vectors_list = [[0.0, 0.0] for _ in range(N)]

    # Precompute neighbor offsets in the 1D index space:
    #  i = row * nCols + col
    #  top = i - nCols  (row-1, col)
    #  left = i - 1     (row, col-1)
    #  top-left = i - nCols - 1 (row-1, col-1)
    #
    # We'll do this inside the loop with simple 'if' checks on row, col.

    for i in range(N):
        row = i // nCols
        col = i % nCols

        # Gather neighbors that exist
        neighbors = []
        if row > 0:
            neighbors.append(array(block_vectors_list[i - nCols]))  # top
        if col > 0:
            neighbors.append(array(block_vectors_list[i - 1]))  # left
        if row > 0 and col > 0:
            neighbors.append(array(block_vectors_list[i - nCols - 1]))  # top-left

        if neighbors:
            neighbors_tensor = stack(neighbors, axis=0)  # shape (#neighbors, 2)
            pV = median(neighbors_tensor, axis=0)  # shape (2,)
        else:
            pV = zeros((2,))  # default if no neighbors

        # cost_volume[i, :] => shape (nDisp,)
        raw_costs = cost_volume[i, :]

        # Smoothness penalty, vectorized for all displacements
        if cost_method == "sad":
            # L1 penalty
            penalty_disp = sum(abs(all_disp - pV), axis=1) * lamb_scaled  # shape (nDisp,)
        else:
            # L2 penalty
            penalty_disp = sum((all_disp - pV) ** 2, axis=1) * lamb_scaled

        penalty_disp = cast(penalty_disp, raw_costs.dtype)
        total_costs = raw_costs + penalty_disp  # shape (nDisp,)

        # Pick displacement that minimizes total_costs
        idx_min = argmin(total_costs)  # single integer index
        best_disp = all_disp[idx_min]  # shape (2,)

        block_vectors_list[i] = best_disp  # store [dy, dx] in the Python list

    # Convert final 1D list => shape (N, 2), then reshape => (nRows, nCols, 2)
    block_vectors = array(block_vectors_list)
    block_vectors = reshape(block_vectors, (nRows, nCols, 2))

    return block_vectors


def _expand_block_vectors(block_vectors, bH, bW, H, W):
    """
    Repeats (tiles) the block vectors across each (bH, bW) region -> (H, W, 2).
    """
    expanded_vectors = repeat(block_vectors, bH, axis=0)  # shape => (nRows*bH, nCols, 2)
    expanded_vectors = repeat(expanded_vectors, bW, axis=1)  # shape => (nRows*bH, nCols*bW, 2)
    mvf = reshape(expanded_vectors, (H, W, 2))
    return mvf


def block_matching(ref, cur, block_shape=(8, 8), search_radius=4, cost_method="ssd", lamb=0.0):
    """
    Efficient block-matching motion estimation with neighbor-based smoothing.

    This function estimates the motion vector field (MVF) between two consecutive frames
    using block-matching. It extracts blocks from the current frame and searches for
    the best match in the reference frame within a given search radius. Additionally,
    it includes a smoothness penalty by encouraging each block's displacement to be
    near the median of its already-computed neighbors (top, left, top-left).

    Args:
        ref (ndarray or tf.Tensor): Reference (previous) frame, 2D shape (H, W).
        cur (ndarray or tf.Tensor): Current  frame,   2D shape (H, W).
        block_shape (tuple): (bH, bW) size of each block.
        search_radius (int): Radius for block search.
        cost_method (str): "ssd" (Sum of Squared Differences) or "sad" (Sum of Absolute Differences).
        lamb (float): Smoothness weight to penalize large deviations from neighbor median.

    Returns:
        mvf (ndarray or tf.Tensor): The final motion vector field of shape (H, W, 2).
            If a block in 'cur' moves by (dy, dx), the returned 'mvf' contains (-dy, -dx).
    """
    # 1) Dimensions & checks
    H, W, bH, bW, nRows, nCols, N = _check_and_prepare_shapes(cur, block_shape)
    lamb_scaled = lamb * bH * bW

    # 2) Pad reference
    pad_ref = _pad_reference(ref, search_radius)

    # 3) Extract blocks from 'cur'
    cur_blocks_4d = _extract_blocks(cur, block_shape, step_row=bH, step_col=bW)
    cur_blocks = reshape(cur_blocks_4d, (N, bH, bW))  # => shape (N, bH, bW)

    # 4) Build cost volume => (N, nDisp), plus the array of all displacements => (nDisp, 2)
    cost_volume, all_disp = _build_cost_volume(pad_ref, cur_blocks, H, W, block_shape, search_radius, cost_method, dtype_for_disp=cur.dtype)

    # 5) Compute blockwise displacements (with neighbor smoothing)
    block_vectors = _compute_blockwise_disp(cost_volume, all_disp, nRows, nCols, cost_method, lamb_scaled)

    # 6) Expand block vectors => (H, W, 2)
    mvf = _expand_block_vectors(block_vectors, bH, bW, H, W)

    # Return negative of motion vectors
    return -mvf
