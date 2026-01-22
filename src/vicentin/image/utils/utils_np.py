from typing import Optional

import numpy as np
from scipy.ndimage import convolve as scipy_convolve


def convolve(
    img: np.ndarray,
    kernel: np.ndarray | tuple[np.ndarray, np.ndarray],
    padding: str = "same",
    strides: int = 1,
    **kwargs,
):
    if isinstance(kernel, (tuple, list)):
        k_vert, k_horz = [np.array(k) for k in kernel]

        v_shape = (-1, 1, 1) if img.ndim == 3 else (-1, 1)
        h_shape = (1, -1, 1) if img.ndim == 3 else (1, -1)

        output = scipy_convolve(
            img, k_vert.reshape(v_shape), mode=padding, **kwargs
        )
        output = scipy_convolve(
            output, k_horz.reshape(h_shape), mode=padding, **kwargs
        )
    else:
        if img.ndim == 3 and kernel.ndim == 2:
            kernel = kernel[:, :, np.newaxis]
        output = scipy_convolve(img, kernel, mode=padding, **kwargs)

    return output[::strides, ::strides, ...]


def gaussian_filter(img: np.ndarray, sigma: float):
    radius = int(4.0 * sigma + 0.5)
    kernel_size = 2 * radius + 1
    coords = np.arange(kernel_size) - radius
    kernel_1d = np.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    return convolve(img, (kernel_1d, kernel_1d), padding="same")


def img2patches(
    img: np.ndarray, patch_shape: tuple[int, int], step_row: int, step_col: int
):
    H, W = img.shape[:2]
    pH, pW = patch_shape

    n_rows = (H - pH) // step_row + 1
    n_cols = (W - pW) // step_col + 1

    new_shape = (n_rows, n_cols, pH, pW)

    new_strides = (
        img.strides[0] * step_row,
        img.strides[1] * step_col,
        img.strides[0],
        img.strides[1],
    )

    patches = np.lib.stride_tricks.as_strided(
        img, shape=new_shape, strides=new_strides, writeable=False
    )

    return patches.copy()


def get_neighbors(
    img: np.ndarray,
    row: int,
    col: int,
    depth: Optional[int] = None,
    neighborhood: int = 4,
):
    H, W = img.shape[:2]
    L = img.shape[2] if img.ndim == 3 else 1
    k = depth if depth is not None else 0

    if neighborhood == 4:
        moves = np.array(
            [
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ]
        )
    else:
        moves = np.array(
            [
                [i, j, m]
                for i in [-1, 0, 1]
                for j in [-1, 0, 1]
                for m in [-1, 0, 1]
                if not (i == j == m == 0)
            ]
        )

    neighbors = np.array([row, col, k]) + moves

    return [
        tuple(neighbor)
        for neighbor in neighbors[
            (0 <= neighbors[:, 0])
            & (neighbors[:, 0] < H)
            & (0 <= neighbors[:, 1])
            & (neighbors[:, 1] < W)
            & (0 <= neighbors[:, 2])
            & (neighbors[:, 2] < L)
        ]
    ]
