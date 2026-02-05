from typing import Optional
from vicentin.image.utils import convolve, gaussian_filter
from vicentin.image.differentiation import grad

import numpy as np


def horn_schunck(
    img1: np.ndarray,
    img2: np.ndarray,
    u0: Optional[np.ndarray],
    v0: Optional[np.ndarray],
    alpha: float = 100,
    iters: int = 100,
    blur: float = 1,
):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if blur > 0:
        img1 = gaussian_filter(img1, blur)
        img2 = gaussian_filter(img2, blur)

    if u0 is None:
        u = np.zeros_like(img1)
    else:
        u = u0.copy()

    if v0 is None:
        v = np.zeros_like(img1)
    else:
        v = v0.copy()

    # Estimate spatiotemporal derivatives
    fx, fy = grad((img1 + img2) / 2, method="diff", boundary="reflect")
    ft = img2 - img1

    avg_kernel = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]) / 12
    denom = alpha**2 + fx**2 + fy**2

    for _ in range(iters):
        # Compute local averages of the flow vectors using kernel_1
        uAvg = convolve(u, avg_kernel, padding="reflect")
        vAvg = convolve(v, avg_kernel, padding="reflect")

        # Compute flow vectors constrained by its local average and the optical flow constraints
        aux = (uAvg * fx + vAvg * fy + ft) / denom

        u = uAvg - fx * aux
        v = vAvg - fy * aux

    u = np.where(np.isnan(u), np.zeros_like(u), u)
    v = np.where(np.isnan(v), np.zeros_like(v), v)

    mvf = np.stack([v, u], axis=-1)
    return mvf
