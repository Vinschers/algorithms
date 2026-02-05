from typing import Optional
from vicentin.image.utils import convolve, gaussian_filter
from vicentin.image.differentiation import grad

import torch


def horn_schunck(
    img1: torch.Tensor,
    img2: torch.Tensor,
    u0: Optional[torch.Tensor],
    v0: Optional[torch.Tensor],
    alpha: float = 100,
    iters: int = 100,
    blur: float = 1,
):
    img1 = img1.float()
    img2 = img2.float()

    if blur > 0:
        img1 = gaussian_filter(img1, blur)
        img2 = gaussian_filter(img2, blur)

    if u0 is None:
        u = torch.zeros_like(img1)
    else:
        u = u0.clone()

    if v0 is None:
        v = torch.zeros_like(img1)
    else:
        v = v0.clone()

    # Estimate spatiotemporal derivatives
    fx, fy = grad((img1 + img2) / 2, method="diff", boundary="reflect")
    ft = img2 - img1

    avg_kernel = (
        torch.tensor(
            [[1, 2, 1], [2, 0, 2], [1, 2, 1]],
            device=img1.device,
            dtype=img1.dtype,
        )
        / 12
    )
    denom = alpha**2 + fx**2 + fy**2

    for _ in range(iters):
        # Compute local averages of the flow vectors using kernel_1
        uAvg = convolve(u, avg_kernel, padding="reflect")
        vAvg = convolve(v, avg_kernel, padding="reflect")

        # Compute flow vectors constrained by its local average and the optical flow constraints
        aux = (uAvg * fx + vAvg * fy + ft) / denom

        u = uAvg - fx * aux
        v = vAvg - fy * aux

    u = torch.nan_to_num(u)
    v = torch.nan_to_num(v)

    mvf = torch.stack([v, u], dim=-1)
    return mvf
