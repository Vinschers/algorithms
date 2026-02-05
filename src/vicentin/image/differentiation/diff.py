from typing import Any, Optional, Tuple
from vicentin.utils import Dispatcher

disp_finite_diffs = Dispatcher()
disp_sobel = Dispatcher()
disp_grad = Dispatcher()
disp_laplacian = Dispatcher()


try:
    from .diff_np import (
        finite_diffs as finite_diffs_np,
        sobel as sobel_np,
        grad as grad_np,
        laplacian as laplacian_np,
    )

    disp_finite_diffs.register("numpy", finite_diffs_np)
    disp_sobel.register("numpy", sobel_np)
    disp_grad.register("numpy", grad_np)
    disp_laplacian.register("numpy", laplacian_np)
except (ImportError, ModuleNotFoundError):
    pass


try:
    from .diff_torch import (
        finite_diffs as finite_diffs_torch,
        sobel as sobel_torch,
        grad as grad_torch,
        laplacian as laplacian_torch,
    )

    disp_finite_diffs.register("torch", finite_diffs_torch)
    disp_sobel.register("torch", sobel_torch)
    disp_grad.register("torch", grad_torch)
    disp_laplacian.register("torch", laplacian_torch)
except (ImportError, ModuleNotFoundError):
    pass


def finite_diffs(
    img: Any,
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
    backend: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Computes the first-order discrete derivatives of an image using finite
    differences.

    Parameters:
    -----------
    img : Any
        Input image.
    mode : str, optional (default='central')
        'central', 'forward', or 'backward'.
    boundary : str, optional (default='reflect')
        'reflect', 'wrap', 'extend', or None (no padding/valid).
    channels_first : bool (default=False)
        Channel dimension order.
    backend : str, optional (default=None)
        Force backend.

    Returns:
    --------
    dx, dy : Tuple[Any, Any]
    """
    disp_finite_diffs.detect_backend(img, backend)
    img = disp_finite_diffs.cast_values(img)

    return disp_finite_diffs(img, mode, boundary, channels_first)


def sobel(
    img: Any,
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
    backend: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Computes image gradients using the Sobel operator.

    Parameters:
    -----------
    img : Any
        Input image.
    mode : str, optional (default='central')
        'forward' (standard) or 'backward' (flipped kernels).
    boundary : str, optional (default='reflect')
        Padding mode.
    channels_first : bool (default=False)
        Channel dimension order.
    backend : str, optional (default=None)
        Force backend.

    Returns:
    --------
    dx, dy : Tuple[Any, Any]
    """
    disp_sobel.detect_backend(img, backend)
    img = disp_sobel.cast_values(img)

    return disp_sobel(img, mode, boundary, channels_first)


def grad(
    img: Any,
    method: Optional[str] = "diff",
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
    backend: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Calculates the spatial gradient of an image.

    Parameters:
    -----------
    img : Any
        Input image.
    method : str, optional (default='diff')
        'diff' or 'sobel'.
    mode : str, optional (default='central')
        Difference mode passed to solver.
    boundary : str, optional (default='reflect')
        Padding mode.
    channels_first : bool (default=False)
        Channel dimension order.
    backend : str, optional (default=None)
        Force backend.

    Returns:
    --------
    dx, dy : Tuple[Any, Any]
    """
    disp_grad.detect_backend(img, backend)
    img = disp_grad.cast_values(img)

    return disp_grad(img, method, mode, boundary, channels_first)


def laplacian(
    img: Any,
    method: Optional[str] = "direct",
    boundary: Optional[str] = "reflect",
    channels_first: bool = False,
    backend: Optional[str] = None,
) -> Any:
    """
    Computes the Laplacian of an image (8-neighbor by default).

    Parameters:
    -----------
    img : Any
        Input image.
    method : str, optional (default='direct')
        'direct' (convolution) or 'diff' (iterative gradient).
    boundary : str, optional (default='reflect')
        Padding mode.
    channels_first : bool (default=False)
        Channel dimension order.
    backend : str, optional (default=None)
        Force backend.

    Returns:
    --------
    lap : Any
    """
    disp_laplacian.detect_backend(img, backend)
    img = disp_laplacian.cast_values(img)

    return disp_laplacian(img, method, boundary, channels_first)
