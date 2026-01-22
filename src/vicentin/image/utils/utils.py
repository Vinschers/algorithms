from typing import Any, Optional, Tuple
from vicentin.utils import Dispatcher

disp_convolve = Dispatcher()
disp_gaussian = Dispatcher()
disp_img2patches = Dispatcher()
disp_neighbors = Dispatcher()

try:
    from .utils_np import (
        convolve as convolve_np,
        gaussian_filter as gaussian_filter_np,
        img2patches as img2patches_np,
        get_neighbors as get_neighbors_np,
    )

    disp_convolve.register("numpy", convolve_np)
    disp_gaussian.register("numpy", gaussian_filter_np)
    disp_img2patches.register("numpy", img2patches_np)
    disp_neighbors.register("numpy", get_neighbors_np)
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .utils_torch import (
        convolve as convolve_torch,
        gaussian_filter as gaussian_filter_torch,
        img2patches as img2patches_torch,
        get_neighbors as get_neighbors_torch,
    )

    disp_convolve.register("torch", convolve_torch)
    disp_gaussian.register("torch", gaussian_filter_torch)
    disp_img2patches.register("torch", img2patches_torch)
    disp_neighbors.register("torch", get_neighbors_torch)
except (ImportError, ModuleNotFoundError):
    pass


def convolve(
    img: Any,
    kernel: Any | Tuple[Any, Any],
    padding: str = "same",
    strides: int = 1,
    backend: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Performs a 2D convolution on an image using a single kernel or a separable kernel pair.

    Convolution is a fundamental operation where a kernel (filter) is slid across an
    image to extract features or transform the signal. This implementation supports
    standard 2D convolution and separable convolution for efficiency.

    Mathematical Formulation:
    -------------------------
    For an image $I$ and kernel $K$, the output $S$ is:
    $$
    S(i, j) = (I * K)(i, j) = \\sum_m \\sum_n I(i-m, j-n) K(m, n)
    $$
    If a tuple of kernels $(K_v, K_h)$ is provided, the function performs separable
    convolution, which is mathematically equivalent to $I * (K_v * K_h)$ but
    computationally cheaper.

    Complexity Analysis:
    --------------------
    - **Time Complexity:**
        - Standard: $O(N \\cdot K_H \\cdot K_W)$, where $N$ is total pixels and $K$ is kernel size.
        - Separable: $O(N \\cdot (K_H + K_W))$, significantly faster for large kernels.
    - **Backends:**
        - NumPy: Uses `scipy.ndimage.convolve`.
        - PyTorch: Uses `torch.nn.functional.conv2d`.

    Parameters:
    -----------
    img : Any
        Input image. Supported shapes: $(H, W)$, $(C, H, W)$, or $(B, C, H, W)$.
    kernel : Any or Tuple[Any, Any]
        The convolution filter. Can be a 2D array or a tuple of two 1D arrays
        representing vertical and horizontal kernels for separable convolution.
    padding : str (default='same')
        Boundary handling mode. Options include `'same'`, `'reflect'`, `'wrap'`, `'constant'`.
    strides : int (default=1)
        The step size for sliding the kernel.
    backend : str, optional (default=None)
        Force a specific backend.

    Returns:
    --------
    output : Any
        The filtered image.
    """

    disp_convolve.detect_backend(img, backend)
    img, kernel = disp_convolve.cast_values(img, kernel)

    return disp_convolve(img, kernel, padding, strides, **kwargs)


def gaussian_filter(
    img: Any, sigma: float, backend: Optional[str] = None
) -> Any:
    """
    Applies a Gaussian blur to an image.

    The Gaussian filter reduces noise and detail by convolving the image with a
    Gaussian kernel. It acts as a low-pass filter, effectively "smoothing" the signal.

    Mathematical Formulation:
    -------------------------
    The 2D Gaussian distribution is defined as:
    $$
    G(x, y) = \\frac{1}{2\\pi\\sigma^2} e^{-\\frac{x^2 + y^2}{2\\sigma^2}}
    $$
    This function implements the filter as a separable convolution using 1D Gaussian
    kernels to improve performance. The kernel radius is determined by $4\\sigma$.

    Complexity Analysis:
    --------------------
    - **Time Complexity:** $O(N \\cdot \\sigma)$, where $N$ is the number of pixels.
      Due to separability, the complexity scales linearly with the standard deviation.
    - **Backends:**
        - NumPy: Implemented via `convolve` with a generated 1D Gaussian kernel.
        - PyTorch: Uses `torch.nn.functional.conv2d`.

    Parameters:
    -----------
    img : Any
        Input image. Supported shapes: $(H, W)$, $(C, H, W)$, or $(B, C, H, W)$.
    sigma : float
        Standard deviation of the Gaussian kernel. Larger values produce more blur.
    backend : str, optional (default=None)
        Force a specific backend.

    Returns:
    --------
    output : Any
        The smoothed image.
    """

    disp_gaussian.detect_backend(img, backend)
    img = disp_gaussian.cast_values(img)

    return disp_gaussian(img, sigma)


def img2patches(
    img: Any,
    patch_shape: tuple[int, int],
    step_row: int,
    step_col: int,
    backend: Optional[str] = None,
) -> Any:
    """
    Extracts sliding window patches from an image.

    This function transforms an image into a collection of overlapping or
    non-overlapping rectangular blocks. It is commonly used for patch-based
    processing in machine learning and local texture analysis.

    Implementation Note:
    --------------------
    The NumPy backend utilizes `as_strided` to create a view of the original
    memory, making the initial window creation $O(1)$ in terms of memory.
    However, the final result is returned as a copy to ensure memory safety.

    Complexity Analysis:
    --------------------
    - **Time Complexity:** $O(N_{patches} \\cdot pH \\cdot pW)$, dominated by the
      memory copy of the patch data.
    - **Space Complexity:** $O(N_{patches} \\cdot pH \\cdot pW)$ for the output array.

    Parameters:
    -----------
    img : Any
        Input image of shape $(H, W, ...)$.
    patch_shape : tuple[int, int]
        The height and width $(pH, pW)$ of each patch.
    step_row : int
        The stride (step size) between patches in the vertical direction.
    step_col : int
        The stride (step size) between patches in the horizontal direction.
    backend : str, optional (default=None)
        Force a specific backend.

    Returns:
    --------
    patches : Any
        A 4D structure containing the extracted patches, typically of shape
        $(N_{rows}, N_{cols}, pH, pW)$.
    """

    disp_gaussian.detect_backend(img, backend)
    img = disp_img2patches.cast_values(img)

    return disp_img2patches(img, patch_shape, step_row, step_col)


def get_neighbors(
    img: Any,
    row: int,
    col: int,
    depth: Optional[int] = None,
    neighborhood: int = 4,
    backend: Optional[str] = None,
) -> list[tuple]:
    """
    Retrieves the indices of adjacent pixels/voxels in a grid.

    This utility function identifies valid neighboring coordinates around a target
    point $(row, col, depth)$, respecting the image boundaries.

    Neighborhood Types:
    -------------------
    - **4-Neighborhood (6 in 3D):** Only considers direct horizontal, vertical,
      and depth-wise adjacencies (Manhattan distance = 1).
    - **8-Neighborhood (26 in 3D):** Includes diagonal adjacencies
      (Chebyshev distance = 1).

    Complexity Analysis:
    --------------------
    - **Time Complexity:** $O(1)$ (constant time), as it only evaluates a
      fixed number of offsets (6 or 26) regardless of image size.
    - **Space Complexity:** $O(1)$ for the returned list of coordinates.

    Parameters:
    -----------
    img : Any
        The reference image used to determine boundary limits.
    row : int
        The vertical coordinate of the target pixel.
    col : int
        The horizontal coordinate of the target pixel.
    depth : int, optional (default=None)
        The depth (channel/slice) coordinate. If None, defaults to 0.
    neighborhood : int (default=4)
        The connectivity type. 4 corresponds to orthogonal neighbors;
        any other value defaults to full connectivity (8/26).
    backend : str, optional (default=None)
        Force a specific backend.

    Returns:
    --------
    neighbors : list[tuple]
        A list of tuples $(r, c, k)$ representing the valid neighboring indices.
    """

    disp_neighbors.detect_backend(img, backend)
    img = disp_neighbors.cast_values(img)

    return disp_neighbors(img, row, col, depth, neighborhood)
