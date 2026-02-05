from typing import Optional, Callable, Tuple

import math
import numpy as np
from scipy import ndimage


def pad_image(
    image: np.ndarray,
    padding: tuple | list,
    mode: Optional[str | float | int] = None,
    channels_first: bool = False,
) -> np.ndarray:

    if mode is None:
        return image

    np_mode = "constant"
    constant_val = 0

    if isinstance(mode, (float, int)):
        np_mode = "constant"
        constant_val = mode
    elif isinstance(mode, str):
        mode_map = {
            "constant": "constant",
            "reflect": "reflect",
            "wrap": "wrap",
            "extend": "edge",
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode '{mode}'.")
        np_mode = mode_map[mode]

    spatial_pads = []

    if len(padding) == 4:
        t, b, l, r = padding
        spatial_pads = [(t, b), (l, r)]
    elif len(padding) == 6:
        f, bk, t, b, l, r = padding
        spatial_pads = [(f, bk), (t, b), (l, r)]
    else:
        raise ValueError(
            "Padding tuple must be length 4 (H, W) or 6 (D, H, W)."
        )

    ndim = image.ndim
    num_spatial = len(spatial_pads)

    final_pads = []

    if ndim == num_spatial:
        final_pads = spatial_pads

    elif ndim == num_spatial + 1:
        if channels_first:
            final_pads = [(0, 0)] + spatial_pads
        else:
            final_pads = spatial_pads + [(0, 0)]

    else:
        raise ValueError(
            f"Shape {image.shape} incompatible with {num_spatial}D padding spec."
        )

    if np_mode == "constant":
        return np.pad(
            image, final_pads, mode=np_mode, constant_values=constant_val
        )

    return np.pad(  # pyright: ignore[reportCallIssue]
        image, final_pads, mode=np_mode  # pyright: ignore[reportArgumentType]
    )


def _prepare_input(
    img: np.ndarray, channels_first: bool
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:

    orig_ndim = img.ndim
    if orig_ndim == 2:
        normalized = img[np.newaxis, :, :]
        restore = lambda x: x.squeeze(0) if x.ndim == 3 else x
    elif orig_ndim == 3:
        if channels_first:
            normalized = img
            restore = lambda x: x
        else:
            normalized = np.transpose(img, (2, 0, 1))
            restore = lambda x: (
                np.transpose(x, (1, 2, 0)) if x.ndim == 3 else x
            )
    else:
        normalized = img
        restore = lambda x: x
    return normalized.astype(np.float32), restore


def _get_valid_slice(
    dim_len: int,
    k_size: int,
    stride: int,
    pad_before: int,
    padding_is_none: bool,
) -> slice:
    if k_size > dim_len:
        return slice(0, 0, 1)

    if padding_is_none:
        start = k_size // 2
    else:
        start = pad_before

    out_len = (dim_len - k_size) // stride + 1
    end = start + (out_len - 1) * stride + 1

    return slice(start, end, stride)


def _parse_strides(
    stride: int | Tuple[int, ...], stride_channels: bool
) -> Tuple[int, int, int]:
    if isinstance(stride, int):
        return (
            (stride, stride, stride) if stride_channels else (stride, stride, 1)
        )

    if len(stride) == 2:
        return stride[0], stride[1], 1

    elif len(stride) == 3:
        return stride[0], stride[1], stride[2]

    elif len(stride) == 1:
        s = stride[0]
        return (s, s, s) if stride_channels else (s, s, 1)

    else:
        raise ValueError(
            f"Stride must be an integer or a tuple of length 1, 2, or 3. Got {stride}"
        )


def _correlate_separable(
    x: np.ndarray,
    kernels: Tuple[np.ndarray, ...],
    strides: Tuple[int, int, int],
    channels_first: bool,
    offsets: Tuple[int, int, int],
    padding_is_none: bool,
) -> np.ndarray:
    s_h, s_w, s_c = strides
    pad_c, pad_h, pad_w = offsets
    out = x

    def apply_1d(data, kernel, axis, stride, pad_amt):
        k = np.atleast_1d(kernel)
        res = ndimage.correlate1d(data, k, axis=axis, mode="constant", cval=0.0)

        dim_len = data.shape[axis]
        sl = _get_valid_slice(
            dim_len, k.shape[0], stride, pad_amt, padding_is_none
        )

        slices = [slice(None)] * data.ndim
        slices[axis] = sl
        return res[tuple(slices)]

    if len(kernels) == 3:
        k_d = kernels[0] if channels_first else kernels[2]
        out = apply_1d(out, k_d, 0, s_c, pad_c)

        k_h = kernels[1] if channels_first else kernels[0]
        k_w = kernels[2] if channels_first else kernels[1]
    else:
        k_h = kernels[0]
        k_w = kernels[1]

    out = apply_1d(out, k_h, 1, s_h, pad_h)
    out = apply_1d(out, k_w, 2, s_w, pad_w)

    return out


def _correlate_standard(
    x: np.ndarray,
    kernel: np.ndarray,
    strides: Tuple[int, int, int],
    channels_first: bool,
    offsets: Tuple[int, int, int],
    padding_is_none: bool,
) -> np.ndarray:
    s_h, s_w, s_c = strides
    pad_c, pad_h, pad_w = offsets
    c, h, w = x.shape

    if kernel.ndim == 2:
        k_full = kernel[np.newaxis, :, :]
    else:
        if channels_first:
            k_full = kernel
        else:
            k_full = kernel.transpose(2, 0, 1)

    out = ndimage.correlate(x, k_full, mode="constant", cval=0.0)

    k_depth = k_full.shape[0]
    if k_depth == c and k_depth > 1:
        center_idx = c // 2
        sl_c = slice(center_idx, center_idx + 1, 1)
    elif kernel.ndim == 2:
        sl_c = slice(None, None, s_c)
    else:
        sl_c = _get_valid_slice(c, k_depth, s_c, pad_c, padding_is_none)

    sl_h = _get_valid_slice(h, k_full.shape[1], s_h, pad_h, padding_is_none)
    sl_w = _get_valid_slice(w, k_full.shape[2], s_w, pad_w, padding_is_none)

    return out[sl_c, sl_h, sl_w]


def _calculate_same_padding(
    in_dim: int, kernel_dim: int, stride: int
) -> Tuple[int, int]:
    out_dim = math.ceil(in_dim / stride)
    needed_input = (out_dim - 1) * stride + kernel_dim
    total_padding = max(0, needed_input - in_dim)

    pad_before = total_padding // 2
    pad_after = total_padding - pad_before

    return pad_before, pad_after


def _pad_input(
    x: np.ndarray,
    kernel: np.ndarray | Tuple[np.ndarray, ...],
    strides: Tuple[int, int, int],
    padding: Optional[str | float | int],
    pad_channels: bool,
    channels_first: bool,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    if padding is None:
        return x, (0, 0, 0)

    c, h, w = x.shape
    s_h, s_w, s_c = strides

    if isinstance(kernel, (list, tuple)):
        if channels_first and len(kernel) == 3:
            k_c, k_h, k_w = kernel[0].size, kernel[1].size, kernel[2].size
        else:
            k_h = kernel[0].size
            k_w = kernel[1].size
            k_c = kernel[2].size if len(kernel) == 3 else 1
    else:
        if kernel.ndim == 2:
            k_h, k_w, k_c = kernel.shape[0], kernel.shape[1], 1
        elif channels_first:
            k_c, k_h, k_w = kernel.shape[0], kernel.shape[1], kernel.shape[2]
        else:
            k_h, k_w, k_c = kernel.shape[0], kernel.shape[1], kernel.shape[2]

    pad_h_before, pad_h_after = _calculate_same_padding(h, k_h, s_h)
    pad_w_before, pad_w_after = _calculate_same_padding(w, k_w, s_w)

    is_volumetric = (k_c > 1 and pad_channels) or (
        channels_first and pad_channels
    )

    if is_volumetric:
        pad_c_before, pad_c_after = _calculate_same_padding(c, k_c, s_c)
        pads_tuple = (
            pad_c_before,
            pad_c_after,
            pad_h_before,
            pad_h_after,
            pad_w_before,
            pad_w_after,
        )
        use_cf = False
        offsets = (pad_c_before, pad_h_before, pad_w_before)
    else:
        pads_tuple = (pad_h_before, pad_h_after, pad_w_before, pad_w_after)
        use_cf = True
        offsets = (0, pad_h_before, pad_w_before)

    pad_mode = "constant" if padding == "same" else padding

    return (
        pad_image(x, pads_tuple, mode=pad_mode, channels_first=use_cf),
        offsets,
    )


def correlate(
    img: np.ndarray,
    kernel: np.ndarray | Tuple[np.ndarray, ...],
    channels_first: bool = False,
    stride: int | Tuple[int, int] = 1,
    stride_channels: bool = False,
    padding: Optional[str | float | int] = "same",
    pad_channels: bool = False,
) -> np.ndarray:

    x, restore_fn = _prepare_input(np.asarray(img), channels_first)

    is_separable = isinstance(kernel, (tuple, list))

    strides = _parse_strides(stride, stride_channels)

    x_padded, offsets = _pad_input(
        x, kernel, strides, padding, pad_channels, channels_first
    )

    padding_is_none = padding is None

    if is_separable:
        output = _correlate_separable(
            x_padded, kernel, strides, channels_first, offsets, padding_is_none
        )
    else:
        output = _correlate_standard(
            x_padded, kernel, strides, channels_first, offsets, padding_is_none
        )

    return restore_fn(output)


def convolve(
    img: np.ndarray,
    kernel: np.ndarray | Tuple[np.ndarray, ...],
    channels_first: bool = False,
    stride: int | Tuple[int, int] = 1,
    stride_channels: bool = False,
    padding: str | float | int = "same",
    pad_channels: bool = False,
) -> np.ndarray:

    if isinstance(kernel, (tuple, list)):
        flipped_kernel = tuple(np.flip(k) for k in kernel)
    else:
        if pad_channels:
            flipped_kernel = np.flip(kernel)
        else:
            if kernel.ndim == 3:
                flipped_kernel = np.flip(kernel, axis=(0, 1))
            else:
                flipped_kernel = np.flip(kernel)

    return correlate(
        img,
        flipped_kernel,
        channels_first,
        stride,
        stride_channels,
        padding,
        pad_channels,
    )


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
