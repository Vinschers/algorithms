from typing import Tuple, Optional

import numpy as np
from vicentin.image.utils import correlate


def _standardize(
    img: np.ndarray, channels_first: bool = False
) -> Tuple[np.ndarray, bool, Tuple[int, ...]]:

    img = np.asarray(img)

    new_img = img
    is_2d = False
    original_shape = img.shape

    if img.ndim == 2:
        new_img = img[..., np.newaxis]
        is_2d = True

    elif img.ndim == 3 and channels_first:
        new_img = np.transpose(img, (1, 2, 0))

    return new_img, is_2d, original_shape


def _restore(img: np.ndarray, is_2d: bool, channels_first: bool) -> np.ndarray:
    if is_2d:
        if img.ndim == 3 and img.shape[-1] == 1:
            return img[..., 0]
        return img

    if channels_first:
        return np.transpose(img, (2, 0, 1))

    return img


def finite_diffs(
    img: np.ndarray,
    mode: str = "central",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    data, is_2d, _ = _standardize(img, channels_first)

    if boundary is not None:
        mode_map = {
            "reflect": "reflect",
            "wrap": "wrap",
            "extend": "edge",
            "constant": "constant",
            "pad": "constant",
        }

        np_mode = mode_map.get(boundary, boundary)

        pad_width = ((1, 1), (1, 1), (0, 0))
        padded = np.pad(  # pyright: ignore[reportCallIssue]
            data, pad_width, mode=np_mode  # pyright: ignore[reportArgumentType]
        )
    else:
        padded = data

    if mode == "central":
        if boundary is not None:
            out_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2
            out_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2
        else:
            out_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2
            out_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2

    elif mode == "forward":
        if boundary is not None:
            out_x = padded[1:-1, 2:] - padded[1:-1, 1:-1]
            out_y = padded[2:, 1:-1] - padded[1:-1, 1:-1]
        else:
            out_x = padded[:, 1:] - padded[:, :-1]
            out_y = padded[1:, :] - padded[:-1, :]

    elif mode == "backward":
        if boundary is not None:
            out_x = padded[1:-1, 1:-1] - padded[1:-1, :-2]
            out_y = padded[1:-1, 1:-1] - padded[:-2, 1:-1]
        else:
            out_x = padded[:, 1:] - padded[:, :-1]
            out_y = padded[1:, :] - padded[:-1, :]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return _restore(out_x, is_2d, channels_first), _restore(
        out_y, is_2d, channels_first
    )


def sobel(
    img: np.ndarray,
    mode: str = "forward",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    smooth = np.array([1, 2, 1], dtype=np.float32)
    diff = np.array([-1, 0, 1], dtype=np.float32)

    if mode == "backward":
        diff = np.flip(diff)

    k_dx = (smooth, diff)
    k_dy = (diff, smooth)

    pad_arg = boundary
    if boundary == "pad":
        pad_arg = "constant"

    dx = correlate(img, k_dx, channels_first, padding=pad_arg)
    dy = correlate(img, k_dy, channels_first, padding=pad_arg)

    return dx, dy


def grad(
    img: np.ndarray,
    method: str = "diff",
    mode: str = "central",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    if method == "diff":
        dx, dy = finite_diffs(img, mode, boundary, channels_first)
    elif method == "sobel":
        dx, dy = sobel(img, mode, boundary, channels_first)
    else:
        raise ValueError(f"Unknown gradient method: {method}")

    return dx, dy


def laplacian(
    img: np.ndarray,
    method: str = "direct",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
) -> np.ndarray:

    if method == "direct":
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=img.dtype)
        res = correlate(img, kernel, channels_first, padding=boundary)

    else:
        dx, dy = grad(img, method, "backward", boundary, channels_first)
        lap_x = grad(dx, method, "forward", boundary, channels_first)[0]
        lap_y = grad(dy, method, "forward", boundary, channels_first)[1]
        res = lap_x + lap_y

    return res
