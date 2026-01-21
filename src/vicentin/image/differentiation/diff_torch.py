from typing import Optional, Tuple, Callable
import functools

import torch
import torch.nn.functional as F


def handle_batch_dims(func: Callable):
    @functools.wraps(func)
    def wrapper(img: torch.Tensor, *args, **kwargs):
        img = img.float()
        orig_ndim = img.ndim

        if orig_ndim == 2:
            img_4d = img.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 3:
            img_4d = img.unsqueeze(0)
        else:
            img_4d = img

        result = func(img_4d, *args, **kwargs)

        def restore(t: torch.Tensor):
            if orig_ndim == 2:
                return t.squeeze(0).squeeze(0)
            if orig_ndim == 3:
                return t.squeeze(0)
            return t

        if isinstance(result, tuple):
            return tuple(restore(res) for res in result)
        return restore(result)

    return wrapper


def _pad_img(img, boundary):
    pad_map = {
        "wrap": "circular",
        "reflect": "reflect",
        "pad": "constant",
        "extend": "replicate",
    }

    img_pad = F.pad(img, (1, 1, 1, 1), mode=pad_map[boundary])
    return img_pad


@handle_batch_dims
def finite_diffs(
    img: torch.Tensor,
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
) -> Tuple[torch.Tensor, torch.Tensor]:

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
    img: torch.Tensor,
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
) -> Tuple[torch.Tensor, torch.Tensor]:

    if mode is None:
        mode = "central"

    channels = img.shape[1]
    img_pad = _pad_img(img, boundary)

    device = img.device
    dtype = img.dtype

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype
    )

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype
    )

    if mode == "backward":
        sobel_x, sobel_y = sobel_x.flip([0, 1]), sobel_y.flip([0, 1])

    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    dx = F.conv2d(img_pad, sobel_x, groups=channels)
    dy = F.conv2d(img_pad, sobel_y, groups=channels)

    return dx, dy


@handle_batch_dims
def grad(
    img: torch.Tensor,
    method: Optional[str] = "diff",
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
) -> Tuple[torch.Tensor, torch.Tensor]:

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
    img: torch.Tensor,
    method: Optional[str] = None,
    boundary: Optional[str] = "reflect",
) -> torch.Tensor:

    if method == "direct":
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            device=img.device,
            dtype=img.dtype,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)

        img_pad = _pad_img(img, boundary)
        return F.conv2d(img_pad, kernel, groups=img.shape[1])

    else:
        img_pad = _pad_img(img, boundary)

        dx, dy = grad(img_pad, method, "backward", "pad")

        lap_x = grad(dx, method, "forward", "pad")[0]
        lap_y = grad(dy, method, "forward", "pad")[1]

        res = lap_x + lap_y
        return res[..., 1:-1, 1:-1]
