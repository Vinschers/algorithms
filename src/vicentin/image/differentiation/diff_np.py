from typing import Optional, Tuple, Callable
import functools

import numpy as np
from scipy.ndimage import correlate


def handle_batch_dims(func: Callable):
    @functools.wraps(func)
    def wrapper(img: np.ndarray, *args, **kwargs):
        if img.dtype not in [np.float32, np.float64]:
            img = img.astype(np.float32)

        orig_ndim = img.ndim

        if orig_ndim == 2:
            img_4d = img[None, None, ...]
        elif orig_ndim == 3:
            img_4d = img[None, ...]
        else:
            img_4d = img

        result = func(img_4d, *args, **kwargs)

        def restore(t: np.ndarray):
            if orig_ndim == 2:
                return t.squeeze(0).squeeze(0)
            if orig_ndim == 3:
                return t.squeeze(0)
            return t

        if isinstance(result, tuple):
            return tuple(restore(res) for res in result)
        return restore(result)

    return wrapper


def _pad_img(img: np.ndarray, boundary) -> np.ndarray:
    pad_map = {
        "wrap": "wrap",
        "reflect": "reflect",
        "pad": "constant",
        "extend": "edge",
    }

    pad_width = [(0, 0), (0, 0), (1, 1), (1, 1)]
    return np.pad(img, pad_width, mode=pad_map[boundary])


@handle_batch_dims
def finite_diffs(
    img: np.ndarray,
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
) -> Tuple[np.ndarray, np.ndarray]:

    if mode is None:
        mode = "central"

    img_pad = _pad_img(img, boundary)

    if mode == "forward":
        dx = img_pad[..., 1:-1, 2:] - img
        dy = img_pad[..., 2:, 1:-1] - img

    elif mode == "backward":
        dx = img - img_pad[..., 1:-1, :-2]
        dy = img - img_pad[..., :-2, 1:-1]

    elif mode == "central":
        dx = (img_pad[..., 1:-1, 2:] - img_pad[..., 1:-1, :-2]) / 2
        dy = (img_pad[..., 2:, 1:-1] - img_pad[..., :-2, 1:-1]) / 2

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return dx, dy


@handle_batch_dims
def sobel(
    img: np.ndarray,
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
) -> Tuple[np.ndarray, np.ndarray]:

    if mode is None:
        mode = "central"

    img_pad = _pad_img(img, boundary)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype)

    if mode == "backward":
        sobel_x = np.flip(sobel_x, axis=(0, 1))
        sobel_y = np.flip(sobel_y, axis=(0, 1))

    sobel_x = sobel_x.reshape(1, 1, 3, 3)
    sobel_y = sobel_y.reshape(1, 1, 3, 3)

    dx = correlate(img_pad, sobel_x, mode="constant", cval=0.0)
    dy = correlate(img_pad, sobel_y, mode="constant", cval=0.0)

    return dx[..., 1:-1, 1:-1], dy[..., 1:-1, 1:-1]


@handle_batch_dims
def grad(
    img: np.ndarray,
    method: Optional[str] = "diff",
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
) -> Tuple[np.ndarray, np.ndarray]:

    if method is None:
        method = "diff"
    if boundary is None:
        boundary = "reflect"

    if method == "diff":
        dx, dy = finite_diffs(img, mode, boundary)
    elif method == "sobel":
        dx, dy = sobel(img, mode, boundary)
    else:
        raise ValueError(f"Unknown gradient method: {method}")

    return dx, dy


@handle_batch_dims
def laplacian(
    img: np.ndarray,
    method: Optional[str] = None,
    boundary: Optional[str] = "reflect",
) -> np.ndarray:

    if method == "direct":
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=img.dtype)
        kernel = kernel.reshape(1, 1, 3, 3)

        img_pad = _pad_img(img, boundary)
        res = correlate(img_pad, kernel, mode="constant", cval=0.0)

    else:
        img_pad = _pad_img(img, boundary)

        dx, dy = grad(img_pad, method, "backward", "pad")

        lap_x = grad(dx, method, "forward", "pad")[0]
        lap_y = grad(dy, method, "forward", "pad")[1]

        res = lap_x + lap_y

    return res[..., 1:-1, 1:-1]
