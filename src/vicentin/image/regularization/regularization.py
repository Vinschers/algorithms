from typing import Any, Optional

from vicentin.utils import Dispatcher


disp_tychonov = Dispatcher()
disp_tv = Dispatcher()

try:
    from .regularization_np import (
        tychonov as tychonov_np,
        total_variation as total_variation_np,
    )

    disp_tychonov.register("numpy", tychonov_np)
    disp_tv.register("numpy", total_variation_np)
except (ImportError, ModuleNotFoundError):
    pass


try:
    from .regularization_torch import (
        tychonov as tychonov_torch,
        total_variation as total_variation_torch,
    )

    disp_tychonov.register("torch", tychonov_torch)
    disp_tv.register("torch", total_variation_torch)
except (ImportError, ModuleNotFoundError):
    pass


def tychonov(img: Any, backend: Optional[str] = None):
    disp_tychonov.detect_backend(img, backend)
    img = disp_tychonov.cast_values(img)

    return disp_tychonov(img)


def total_variation(
    img: Any, epsilon: float = 1e-2, backend: Optional[str] = None
):
    disp_tv.detect_backend(img, backend)
    img = disp_tv.cast_values(img)

    return disp_tv(img, epsilon)
