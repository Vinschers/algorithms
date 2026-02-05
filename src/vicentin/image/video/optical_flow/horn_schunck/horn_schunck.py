from typing import Any, Optional
from vicentin.utils import Dispatcher

disp_hs = Dispatcher()

try:
    from .horn_schunck_np import horn_schunck as horn_schunck_np

    disp_hs.register("numpy", horn_schunck_np)
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .horn_schunck_torch import horn_schunck as horn_schunck_torch

    disp_hs.register("torch", horn_schunck_torch)
except (ImportError, ModuleNotFoundError):
    pass


def horn_schunck(
    img1: Any,
    img2: Any,
    u0: Optional[Any] = None,
    v0: Optional[Any] = None,
    alpha: float = 100,
    iters: int = 100,
    blur: float = 1,
    backend: Optional[str] = None,
):
    """
    Computes the optical flow between two images using the Horn-Schunck method.

    This method is based on the original paper:
    Horn, B.K.P., and Schunck, B.G., "Determining Optical Flow,"
    Artificial Intelligence, Vol. 17, No. 1-3, August 1981, pp. 185-203.
    [Link: http://dspace.mit.edu/handle/1721.1/6337]

    Parameters
    ----------
    img1 : np.array
        First image (grayscale or single-channel frame).
    img2 : np.array
        Second image (grayscale or single-channel frame), captured after `img1`.
    u0 : np.array
        Initial horizontal flow estimate.
    v0 : np.array
        Initial vertical flow estimate.
    alpha : float, optional (default=100)
        Regularization parameter controlling the smoothness of the flow.
    iters : int, optional (default=100)
        Number of iterations to perform.
    blur : float, optional (default=1)
        Standard deviation for Gaussian smoothing applied to the images before computing derivatives.
    backend : str, optional (default=None)
        Force a specific backend.

    Returns
    -------
    mvf : np.array, shape (H, W, 2)
        Estimated motion vector field, where:
        - `mvf[..., 0]` contains the vertical flow component (v).
        - `mvf[..., 1]` contains the horizontal flow component (u).

    Notes
    -----
    - The algorithm estimates the optical flow by enforcing brightness constancy and a smoothness constraint.
    - A weighted 3x3 averaging kernel is used to iteratively refine the flow estimates.
    - Gaussian smoothing is applied to reduce noise in derivative computation.
    - If initial flow estimates (`u0`, `v0`) are good, convergence is faster.
    """

    disp_hs.detect_backend(img1, backend)

    args = [img1, img2]
    if u0 is not None:
        args.append(u0)
    if v0 is not None:
        args.append(v0)

    casted_args = disp_hs.cast_values(*args)

    img1, img2 = casted_args[0], casted_args[1]
    u0 = casted_args[2] if u0 is not None else None
    v0 = casted_args[3] if v0 is not None else None

    return disp_hs(img1, img2, u0, v0, alpha, iters, blur)
