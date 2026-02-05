from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from vicentin.image.utils import correlate


def _standardize(
    img: torch.Tensor, channels_first: bool = False
) -> Tuple[torch.Tensor, bool, Tuple[int, ...]]:

    new_img = img
    is_2d = False
    original_shape = tuple(img.shape)

    if img.ndim == 2:
        new_img = img.unsqueeze(-1)
        is_2d = True

    elif img.ndim == 3 and channels_first:
        new_img = img.permute(1, 2, 0)

    return new_img, is_2d, original_shape


def _restore(
    img: torch.Tensor, is_2d: bool, channels_first: bool
) -> torch.Tensor:
    if is_2d:
        return img.squeeze(-1)

    if channels_first:
        return img.permute(2, 0, 1)

    return img


def finite_diffs(
    img: torch.Tensor,
    mode: str = "central",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    data, is_2d, _ = _standardize(img, channels_first)

    if boundary is not None:
        data_chw = data.permute(2, 0, 1)

        mode_map = {
            "reflect": "reflect",
            "wrap": "circular",
            "extend": "replicate",
            "constant": "constant",
            "pad": "constant",
        }

        if boundary not in mode_map:
            pt_mode = boundary
        else:
            pt_mode = mode_map[boundary]

        pad_width = (1, 1, 1, 1)

        needs_unsqueeze = pt_mode in ["circular", "replicate", "reflect"]

        if needs_unsqueeze and data_chw.ndim == 3:
            data_chw = data_chw.unsqueeze(0)

        padded_chw = F.pad(data_chw, pad_width, mode=pt_mode)

        if needs_unsqueeze and data_chw.ndim == 4:
            padded_chw = padded_chw.squeeze(0)

        padded = padded_chw.permute(1, 2, 0)
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
    img: torch.Tensor,
    mode: str = "forward",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = img.device
    dtype = img.dtype

    smooth = torch.tensor([1.0, 2.0, 1.0], dtype=dtype, device=device)
    diff = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype, device=device)

    if mode == "backward":
        diff = torch.flip(diff, dims=(0,))

    k_dx = (smooth, diff)
    k_dy = (diff, smooth)

    pad_arg = boundary
    if boundary == "pad":
        pad_arg = "constant"

    dx = correlate(img, k_dx, channels_first, padding=pad_arg)
    dy = correlate(img, k_dy, channels_first, padding=pad_arg)

    return dx, dy


def grad(
    img: torch.Tensor,
    method: str = "diff",
    mode: str = "central",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if method == "diff":
        dx, dy = finite_diffs(img, mode, boundary, channels_first)
    elif method == "sobel":
        dx, dy = sobel(img, mode, boundary, channels_first)
    else:
        raise ValueError(f"Unknown gradient method: {method}")

    return dx, dy


def laplacian(
    img: torch.Tensor,
    method: str = "direct",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
) -> torch.Tensor:

    if method == "direct":
        kernel = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=img.dtype,
            device=img.device,
        )
        res = correlate(img, kernel, channels_first, padding=boundary)

    else:
        dx, dy = grad(img, method, "backward", boundary, channels_first)
        lap_x = grad(dx, method, "forward", boundary, channels_first)[0]
        lap_y = grad(dy, method, "forward", boundary, channels_first)[1]
        res = lap_x + lap_y

    return res
