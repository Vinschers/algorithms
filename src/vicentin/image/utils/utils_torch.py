import torch
import torch.nn.functional as F

from typing import Optional


def convolve(
    img: torch.Tensor,
    kernel: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    padding: str = "same",
    strides: int = 1,
):
    if img.ndim == 2:
        img_t = img.unsqueeze(0).unsqueeze(0)
    else:
        img_t = img.permute(2, 0, 1).unsqueeze(0)

    channels = img_t.shape[1]

    if isinstance(kernel, (tuple, list)):
        k_vert, k_horz = kernel

        k_vert = k_vert.view(channels, 1, -1, 1)
        k_horz = k_horz.view(channels, 1, 1, -1)

        output = F.conv2d(
            img_t, k_vert, stride=(strides, 1), padding=padding, groups=channels
        )
        output = F.conv2d(
            output,
            k_horz,
            stride=(1, strides),
            padding=padding,
            groups=channels,
        )

    else:
        if kernel.ndim == 2:
            k = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        else:
            k = kernel

        output = F.conv2d(
            img_t, k, stride=strides, padding=padding, groups=channels
        )

    output = output.squeeze(0)

    if img.ndim == 3:
        return output.permute(1, 2, 0)

    return output.squeeze(0)


def gaussian_filter(img: torch.Tensor, sigma: float) -> torch.Tensor:
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
