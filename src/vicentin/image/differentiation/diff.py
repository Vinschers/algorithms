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
    backend: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Computes the first-order discrete derivatives of an image using finite
    differences.

    This function approximates the gradient vector
    $\\nabla I = (\\frac{\\partial I}{\\partial x}, \\frac{\\partial I}{\\partial y})$
    by computing discrete differences between adjacent pixels along the last two dimensions (H, W).

    Mathematical Formulation:
    -------------------------
    The output depends on the chosen `mode`. For a 1D signal $f[i]$:
    - **Forward:** $\\Delta f[i] = f[i+1] - f[i]$
    - **Backward:** $\\Delta f[i] = f[i] - f[i-1]$
    - **Central:** $\\Delta f[i] = \\frac{f[i+1] - f[i-1]}{2}$


    Boundary Handling:
    ------------------
    Images are padded prior to difference computation to maintain input spatial dimensions.
    The padding method is controlled by the `boundary` argument (e.g., 'reflect', 'wrap').

    Complexity Analysis:
    --------------------
    - **Time Complexity:** $O(N)$, where $N$ is the total number of pixels. The operation involves
      a constant number of subtractions per pixel.
    - **Space Complexity:** $O(N)$ for storing the output gradients $d_x$ and $d_y$.

    Parameters:
    -----------
    img : Any (Tensor or ndarray)
        Input image. The function automatically handles the following shapes:
        - 2D: $(H, W)$
        - 3D: $(C, H, W)$
        - 4D: $(B, C, H, W)$
    mode : str, optional (default='central')
        The difference scheme to use. Options:
        - `'central'`: More accurate ($O(h^2)$ error), unbiased.
        - `'forward'`: Standard forward difference ($O(h)$ error).
        - `'backward'`: Standard backward difference ($O(h)$ error).
    boundary : str, optional (default='reflect')
        Padding mode for boundary conditions. Options:
        - `'reflect'`: Reflection padding (e.g., `cba|abcd|dcb`).
        - `'wrap'`: Circular padding.
        - `'pad'`: Zero padding (constant 0).
        - `'extend'`: Edge replication.
    backend : str, optional (default=None)
        Force a specific backend ('numpy' or 'torch'). If None, automatically detected.

    Returns:
    --------
    dx, dy : Tuple[Any, Any]
        Horizontal ($x$) and vertical ($y$) derivative maps.
        The shape matches the input `img`.
    """

    disp_finite_diffs.detect_backend(img, backend)
    img = disp_finite_diffs.cast_values(img)

    return disp_finite_diffs(img, mode=mode, boundary=boundary)


def sobel(
    img: Any,
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
    backend: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Computes image gradients using the Sobel operator.

    The Sobel operator calculates the gradient approximation by convolving the image with
    two $3 \\times 3$ kernels. These kernels emphasize high-frequency changes (edges) while providing
    slight Gaussian smoothing orthogonal to the derivative direction to reduce noise.

    Kernels:
    --------
    For `mode='central'`, the convolution kernels $G_x$ and $G_y$ are:
    $$
    G_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix}, \\quad
    G_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix}
    $$


    If `mode='backward'`, the kernels are flipped relative to the central formulation.

    Complexity Analysis:
    --------------------
    - **Time Complexity:** $O(K \\cdot N)$, where $K=9$ (kernel size) and $N$ is the number of pixels.
      This is effectively linear $O(N)$.
    - **Backends:**
      - NumPy: Uses `scipy.ndimage.correlate`.
      - PyTorch: Uses `torch.nn.functional.conv2d`.

    Parameters:
    -----------
    img : Any (Tensor or ndarray)
        Input image. Supported shapes: $(H, W)$, $(C, H, W)$, or $(B, C, H, W)$.
    mode : str, optional (default='central')
        Kernel orientation configuration.
        - `'central'`: Standard Sobel configuration.
        - `'backward'`: Flips the kernels along both axes.
    boundary : str, optional (default='reflect')
        Padding mode used before convolution to handle image borders.
        Options: `'reflect'`, `'wrap'`, `'pad'` (zero), `'extend'` (edge).
    backend : str, optional (default=None)
        Force a specific backend ('numpy' or 'torch'). If None, automatically detected.

    Returns:
    --------
    dx, dy : Tuple[Any, Any]
        Horizontal and vertical gradient maps.
    """

    disp_sobel.detect_backend(img, backend)
    img = disp_sobel.cast_values(img)

    return disp_sobel(img, mode=mode, boundary=boundary)


def grad(
    img: Any,
    method: Optional[str] = "diff",
    mode: Optional[str] = "central",
    boundary: Optional[str] = "reflect",
    backend: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Calculates the spatial gradient of an image using a specified method.

    This is a high-level wrapper that delegates to either finite differences or
    Sobel convolution. The gradient field $\\nabla I$ points in the direction of
    the greatest rate of increase of the intensity function.


    Methods:
    --------
    1. **'diff'**: Uses simple finite differences. Best for theoretical exactness on
       discrete grids or when no smoothing is desired.
    2. **'sobel'**: Uses Sobel operator. Best for edge detection in noisy images due
       to built-in smoothing.

    Complexity Analysis:
    --------------------
    Both methods operate in $O(N)$ time, though 'sobel' has a higher constant factor due to
    kernel multiplications.

    Parameters:
    -----------
    img : Any
        Input image. Supported shapes: $(H, W)$, $(C, H, W)$, or $(B, C, H, W)$.
    method : str, optional (default='diff')
        The algorithm to use. Options: `'diff'`, `'sobel'`.
    mode : str, optional (default='central')
        The difference mode (`'central'`, `'forward'`, `'backward'`) passed to the underlying solver.
    boundary : str, optional (default='reflect')
        Boundary handling mode (`'reflect'`, `'wrap'`, `'pad'`, `'extend'`).
    backend : str, optional (default=None)
        Force a specific backend.

    Returns:
    --------
    dx, dy : Tuple[Any, Any]
        The calculated gradient components.
    """

    disp_grad.detect_backend(img, backend)
    img = disp_grad.cast_values(img)

    return disp_grad(img, method=method, mode=mode, boundary=boundary)


def laplacian(
    img: Any,
    method: Optional[str] = "direct",
    boundary: Optional[str] = "reflect",
    backend: Optional[str] = None,
) -> Any:
    """
    Computes the Laplacian of an image, measuring the 2nd-order spatial derivatives.

    The Laplacian operator $\\Delta I$ (or $\\nabla^2 I$) highlights regions of rapid intensity change
    and is defined as the divergence of the gradient:
    $$
    \\Delta I = \\frac{\\partial^2 I}{\\partial x^2} + \\frac{\\partial^2 I}{\\partial y^2}
    $$


    Algorithms:
    -----------
    The function supports two computation strategies:

    1. **Direct Convolution (`method='direct'`)**:
       Convolves the image with a standard 8-neighbor isotropic Laplacian kernel:
       $$
       K = \\begin{bmatrix} 1 & 1 & 1 \\\\ 1 & -8 & 1 \\\\ 1 & 1 & 1 \\end{bmatrix}
       $$
    2. **Iterative Gradient (`method='diff'` or others)**:
       Computes the Laplacian by taking the divergence of the gradient.
       It effectively performs:
       $$
       \\Delta I \\approx \\text{div}(\\nabla I) = \\frac{\\partial(\\partial_x I)}{\\partial x} + \\frac{\\partial(\\partial_y I)}{\\partial y}
       $$
       *Note:* To ensure pixel alignment, this path computes inner gradients using `mode='backward'`
       and outer gradients using `mode='forward'`.

    Complexity Analysis:
    --------------------
    - **Direct:** Single convolution pass, $O(K \\cdot N)$. Fastest.
    - **Iterative:** Requires 4 passes (compute dx, dy, then compute dx of dx, dy of dy).
      Slower but mathematically consistent with the `grad` function.

    Parameters:
    -----------
    img : Any
        Input image. Supported shapes: $(H, W)$, $(C, H, W)$, or $(B, C, H, W)$.
    method : str, optional (default=None)
        Strategy to use.
        - `'direct'`: Uses the 8-neighbor hardcoded kernel.
        - `None` (or `'diff'`, `'sobel'`): Delegates to `grad()` recursively.
    boundary : str, optional (default='reflect')
        Boundary handling mode (`'reflect'`, `'wrap'`, `'pad'`, `'extend'`).
    backend : str, optional (default=None)
        Force a specific backend.

    Returns:
    --------
    lap : Any
        The Laplacian map of the input image.
    """

    disp_laplacian.detect_backend(img, backend)
    img = disp_laplacian.cast_values(img)

    return disp_laplacian(img, method=method, boundary=boundary)
