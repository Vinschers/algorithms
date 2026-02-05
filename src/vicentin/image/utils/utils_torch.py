from typing import Optional, Callable, Tuple, Union
import math

import torch
import torch.nn.functional as F


def pad_image(
    image: torch.Tensor,
    padding: Tuple[int, ...],
    mode: Optional[Union[str, float, int]] = None,
    channels_first: bool = False,
) -> torch.Tensor:
    if mode is None:
        return image

    pt_mode = "constant"
    constant_val = 0.0

    if isinstance(mode, (float, int)):
        pt_mode = "constant"
        constant_val = float(mode)
    elif isinstance(mode, str):
        mode_map = {
            "constant": "constant",
            "reflect": "reflect",
            "wrap": "circular",
            "extend": "replicate",
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode '{mode}'.")
        pt_mode = mode_map[mode]

    spatial_pads = []
    if len(padding) == 4:
        t, b, l, r = padding
        spatial_pads = [(l, r), (t, b)]
    elif len(padding) == 6:
        f, bk, t, b, l, r = padding
        spatial_pads = [(l, r), (t, b), (f, bk)]
    else:
        raise ValueError(
            "Padding tuple must be length 4 (H, W) or 6 (D, H, W)."
        )

    ndim = image.ndim
    num_spatial = len(spatial_pads)

    dims_swapped = False
    if not channels_first and ndim == num_spatial + 1:
        perm = (ndim - 1,) + tuple(range(ndim - 1))
        image = image.permute(perm)
        dims_swapped = True

    final_pads = []
    for p in spatial_pads:
        final_pads.extend(p)

    need_unsqueeze = False
    if pt_mode != "constant":
        if len(final_pads) == 6 and image.ndim == 3:
            image = image.unsqueeze(0)
            need_unsqueeze = True
        elif len(final_pads) == 4 and image.ndim == 2:
            image = image.unsqueeze(0)
            need_unsqueeze = True

    padded = F.pad(image, tuple(final_pads), mode=pt_mode, value=constant_val)

    if need_unsqueeze:
        padded = padded.squeeze(0)

    if dims_swapped:
        perm = tuple(range(1, ndim)) + (0,)
        padded = padded.permute(perm)

    return padded


def _prepare_input(
    img: torch.Tensor, channels_first: bool
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:

    img = img.float()

    orig_ndim = img.ndim

    if orig_ndim == 2:
        normalized = img.unsqueeze(0).unsqueeze(0)
        restore = lambda x: x.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        if channels_first:
            normalized = img.unsqueeze(0)
            restore = lambda x: x.squeeze(0)
        else:
            normalized = img.permute(2, 0, 1).unsqueeze(0)
            restore = lambda x: x.squeeze(0).permute(1, 2, 0)
    else:
        normalized = img.unsqueeze(0)
        restore = lambda x: x.squeeze(0)

    return normalized, restore


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
            f"Stride must be int or tuple length 1-3. Got {stride}"
        )


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
    x: torch.Tensor,
    kernel: torch.Tensor | Tuple[torch.Tensor, ...],
    strides: Tuple[int, int, int],
    padding: Optional[Union[str, float, int]],
    pad_channels: bool,
    channels_first: bool,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:

    if padding is None:
        return x, (0, 0, 0)

    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    s_h, s_w, s_c = strides

    if isinstance(kernel, (list, tuple)):
        if channels_first and len(kernel) == 3:
            k_c, k_h, k_w = (
                kernel[0].numel(),
                kernel[1].numel(),
                kernel[2].numel(),
            )
        else:
            k_h = kernel[0].numel()
            k_w = kernel[1].numel()
            k_c = kernel[2].numel() if len(kernel) == 3 else 1
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

    x_padded = pad_image(
        x.squeeze(0), pads_tuple, mode=pad_mode, channels_first=use_cf
    )

    return x_padded.unsqueeze(0), offsets


def _correlate_separable(
    x: torch.Tensor,
    kernels: Tuple[torch.Tensor, ...],
    strides: Tuple[int, int, int],
    channels_first: bool,
    offsets: Tuple[int, int, int],
) -> torch.Tensor:

    s_h, s_w, s_c = strides
    pad_c, pad_h, pad_w = offsets

    if len(kernels) == 3 or (channels_first and len(kernels) == 3):
        k_c_idx = 0 if (channels_first and len(kernels) == 3) else 2
        k_c_tensor = torch.as_tensor(
            kernels[k_c_idx], device=x.device, dtype=x.dtype
        )

        b, c, h, w = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, 1, c)  # (N*H*W, 1, C)

        weight_c = k_c_tensor.view(1, 1, -1)

        out_c = F.conv1d(x_reshaped, weight_c, stride=s_c)

        c_out = out_c.shape[2]
        x = out_c.view(b, h, w, c_out).permute(0, 3, 1, 2)

    elif s_c > 1:
        x = x[:, pad_c::s_c, :, :]

    if channels_first and len(kernels) == 3:
        k_h_tensor = torch.as_tensor(kernels[1], device=x.device, dtype=x.dtype)
        k_w_tensor = torch.as_tensor(kernels[2], device=x.device, dtype=x.dtype)
    else:
        k_h_tensor = torch.as_tensor(kernels[0], device=x.device, dtype=x.dtype)
        k_w_tensor = torch.as_tensor(kernels[1], device=x.device, dtype=x.dtype)

    channels = x.shape[1]

    weight_h = k_h_tensor.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(x, weight_h, stride=(s_h, 1), groups=channels)

    weight_w = k_w_tensor.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    x = F.conv2d(x, weight_w, stride=(1, s_w), groups=channels)

    return x


def _correlate_standard(
    x: torch.Tensor,
    kernel: torch.Tensor,
    strides: Tuple[int, int, int],
    channels_first: bool,
    offsets: Tuple[int, int, int],
) -> torch.Tensor:
    s_h, s_w, s_c = strides
    pad_c, pad_h, pad_w = offsets
    b, c, h, w = x.shape

    kernel = torch.as_tensor(kernel, device=x.device, dtype=x.dtype)

    if kernel.ndim == 2:
        k_h, k_w = kernel.shape
        k_c = 1
        weight = kernel.view(1, 1, k_h, k_w).repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, stride=(s_h, s_w), groups=c)

    else:
        if channels_first:
            k_c, k_h, k_w = kernel.shape
        else:
            k_h, k_w, k_c = kernel.shape
            kernel = kernel.permute(2, 0, 1)

        if k_c == c and k_c > 1:
            weight = kernel.unsqueeze(0)
            return F.conv2d(x, weight, stride=(s_h, s_w))

        else:
            x_vol = x.unsqueeze(1)
            weight = kernel.unsqueeze(0).unsqueeze(0)

            out = F.conv3d(x_vol, weight, stride=(s_c, s_h, s_w))

            return out.squeeze(1)


def correlate(
    img: torch.Tensor,
    kernel: torch.Tensor | Tuple[torch.Tensor, ...],
    channels_first: bool = False,
    stride: int | Tuple[int, ...] = 1,
    stride_channels: bool = False,
    padding: str | float | int = "same",
    pad_channels: bool = False,
) -> torch.Tensor:

    x, restore_fn = _prepare_input(img, channels_first)

    is_separable = isinstance(kernel, (tuple, list))
    strides = _parse_strides(stride, stride_channels)

    x_padded, offsets = _pad_input(
        x, kernel, strides, padding, pad_channels, channels_first
    )

    if is_separable:
        output = _correlate_separable(
            x_padded, kernel, strides, channels_first, offsets
        )
    else:
        output = _correlate_standard(
            x_padded, kernel, strides, channels_first, offsets
        )

    return restore_fn(output)


def convolve(
    img: torch.Tensor,
    kernel: torch.Tensor | Tuple[torch.Tensor, ...],
    channels_first: bool = False,
    stride: int | Tuple[int, ...] = 1,
    stride_channels: bool = False,
    padding: str | float | int = "same",
    pad_channels: bool = False,
) -> torch.Tensor:

    if isinstance(kernel, (tuple, list)):
        flipped_kernel = tuple(
            torch.flip(torch.as_tensor(k), dims=(0,)) for k in kernel
        )
    else:
        k_tsr = torch.as_tensor(kernel)
        if pad_channels:
            dims = tuple(range(k_tsr.ndim))
        else:
            if channels_first:
                dims = (1, 2)
            else:
                dims = (0, 1)

        flipped_kernel = torch.flip(k_tsr, dims=dims)

    return correlate(
        img,
        flipped_kernel,
        channels_first,
        stride,
        stride_channels,
        padding,
        pad_channels,
    )


def gaussian_filter(img: torch.Tensor, sigma: float):
    radius = int(4.0 * sigma + 0.5)
    kernel_size = 2 * radius + 1
    coords = (
        torch.arange(kernel_size, dtype=img.dtype, device=img.device) - radius
    )
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    return convolve(img, (kernel_1d, kernel_1d), padding="same")


def img2patches(
    img: torch.Tensor,
    patch_shape: tuple[int, int],
    step_row: int,
    step_col: int,
) -> torch.Tensor:

    pH, pW = patch_shape
    patches = img.unfold(0, pH, step_row).unfold(1, pW, step_col)
    return patches.clone()


def get_neighbors(
    img: torch.Tensor,
    row: int,
    col: int,
    depth: Optional[int] = None,
    neighborhood: int = 4,
):
    H, W = img.shape[:2]
    L = img.shape[2] if img.ndim == 3 else 1
    k = depth if depth is not None else 0

    if neighborhood == 4:
        moves = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
    else:
        moves = [
            (i, j, m)
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            for m in [-1, 0, 1]
            if not (i == 0 and j == 0 and m == 0)
        ]

    neighbors = []
    for dr, dc, dk in moves:
        r, c, d = row + dr, col + dc, k + dk
        if 0 <= r < H and 0 <= c < W and 0 <= d < L:
            neighbors.append((r, c, d))

    return neighbors
