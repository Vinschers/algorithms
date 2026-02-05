from typing import Any, Optional, Tuple
from vicentin.utils import Dispatcher

disp_pad = Dispatcher()
disp_correlate = Dispatcher()
disp_convolve = Dispatcher()
disp_gaussian = Dispatcher()
disp_img2patches = Dispatcher()
disp_neighbors = Dispatcher()

try:
    from .utils_np import (
        pad_image as pad_image_np,
        convolve as convolve_np,
        correlate as correlate_np,
        gaussian_filter as gaussian_filter_np,
        img2patches as img2patches_np,
        get_neighbors as get_neighbors_np,
    )

    disp_pad.register("numpy", pad_image_np)
    disp_convolve.register("numpy", convolve_np)
    disp_correlate.register("numpy", correlate_np)
    disp_gaussian.register("numpy", gaussian_filter_np)
    disp_img2patches.register("numpy", img2patches_np)
    disp_neighbors.register("numpy", get_neighbors_np)
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .utils_torch import (
        pad_image as pad_image_torch,
        convolve as convolve_torch,
        correlate as correlate_torch,
        gaussian_filter as gaussian_filter_torch,
        img2patches as img2patches_torch,
        get_neighbors as get_neighbors_torch,
    )

    disp_pad.register("torch", pad_image_torch)
    disp_correlate.register("torch", correlate_torch)
    disp_convolve.register("torch", convolve_torch)
    disp_gaussian.register("torch", gaussian_filter_torch)
    disp_img2patches.register("torch", img2patches_torch)
    disp_neighbors.register("torch", get_neighbors_torch)
except (ImportError, ModuleNotFoundError):
    pass


def pad_image(
    image: Any,
    padding: tuple | list,
    mode: Optional[str | float | int] = None,
    channels_first: bool = False,
    backend: Optional[str] = None,
):
    """
    Pads an image array with flexible padding modes and channel ordering.

    Args:
        image (Any): Input image. Supported shapes:
            - 2D: (H, W)
            - 3D: (H, W, C) or (C, H, W)
            - 3D Volumetric: (D, H, W)
        padding (tuple | list): Padding amounts.
            - Length 4: (top, bottom, left, right) -> Applies to H, W
            - Length 6: (front, back, top, bottom, left, right) -> Applies to D, H, W
        mode (str | float | int):
            - If float/int: Pads with this constant value.
            - 'constant': Pads with 0.
            - 'reflect': Mirrors values (e.g., d c b a | a b c d | d c b a).
            - 'wrap': Wraps around (e.g., a b c d | a b c d | a b c d).
            - 'extend': Uses edge values (e.g., a a a a | a b c d | d d d d).
            - None: Returns image unchanged.
        channels_first (bool):
            - True: Channel dim is index 0 (C, H, W) or (C, D, H, W).
            - False: Channel dim is last (H, W, C) or (D, H, W, C).
            - (Note: 2D (H, W) or 3D (D, H, W) inputs without C are handled automatically).
        backend : str, optional (default=None)
            Force a specific backend.

    Returns:
        Any: The padded image.
    """
    disp_pad.detect_backend(image, backend)
    image = disp_pad.cast_values(image)

    return disp_pad(image, padding, mode, channels_first)


def correlate(
    img: Any,
    kernel: Any | Tuple[Any, Any],
    channels_first: bool = False,
    stride: int | Tuple[int, int] = 1,
    stride_channels: bool = False,
    padding: Optional[str | float | int] = "same",
    pad_channels: bool = False,
    backend: Optional[str] = None,
) -> Any:
    """
    Performs a generic correlation on 2D images (spatial) or 3D volumes (volumetric).

    This function unifies standard 2D correlation, CNN-style multi-channel correlation,
    and true 3D volumetric correlation into a single interface. The behavior is
    determined by the shape of the input `kernel` and the `pad_channels` flag.

    Parameters:
    -----------
    img : np.ndarray
        Input data. Can be:
        - 2D: (H, W) or (C, H, W) / (H, W, C)
        - 3D: (D, H, W)
        Note: This function expects a single sample, not a batch of images.
    kernel : np.ndarray | Tuple[np.ndarray, ...]
        The filter to correlate with.
        - For 2D/CNN: shape (kH, kW) or (kH, kW, kCh).
        - For 3D: shape (kD, kH, kW).
        - Separable: A tuple of 1D arrays `(k_vert, k_horz)` or `(k_depth, k_vert, k_horz)`.
    channels_first : bool (default=False)
        If True, the first dimension is treated as the channel/depth dimension.
    stride : int | Tuple[int, int] (default=1)
        Spatial stride (step size).
    stride_channels : bool (default=False)
        If True, the `stride` is also applied to the channel/depth dimension.
    padding : str | float | int | None (default='same')
        Determines padding behavior.
        - **None**: No padding is applied (equivalent to 'valid'). Output size shrinks.
        - **'same'**: Pads with 0 to maintain input spatial dimensions.
        - **'reflect'**: Pads by mirroring values (d c b a | a b c d | d c b a).
        - **'wrap'**: Pads by wrapping around (a b c d | a b c d | a b c d).
        - **'extend'**: Pads using edge values (a a a a | a b c d | d d d d).
        - **float/int**: Pads with this specific constant value.
    pad_channels : bool (default=False)
        **CRITICAL PARAMETER**: Controls the "mode" of operation.
        - `False` (CNN Mode): The channel dimension is NOT padded. Use this when the
          3rd dimension represents color channels (RGB) that should be summed over.
        - `True` (Volumetric Mode): The channel dimension IS padded. Use this when
          the 3rd dimension represents spatial depth (e.g., MRI slices, video frames)
          and you want to slide the kernel across it.
    backend : str, optional (default=None)
        Force a specific backend.

    Returns:
    --------
    output : Any
        The filtered image.

    Examples:
    ---------
    >>> import numpy as np

    **1. Standard 2D Correlation (Grayscale)**
    >>> img = np.random.rand(100, 100)
    >>> kernel = np.ones((3, 3))
    >>> out = correlate(img, kernel)
    >>> # Result: (100, 100) - Standard spatial filtering

    **2. CNN-Style Correlation (RGB Input)**
    >>> # Input: (H=100, W=100, C=3)
    >>> img_rgb = np.random.rand(100, 100, 3)
    >>> # Kernel: (H=3, W=3, C=3) -> Matches input depth
    >>> kernel_cnn = np.ones((3, 3, 3))
    >>> # pad_channels=False ensures we don't pad depth, effectively summing it
    >>> out = correlate(img_rgb, kernel_cnn, pad_channels=False)
    >>> # Result: (100, 100, 1)

    **3. Volumetric Correlation (3D Medical Scan)**
    >>> # Input: (Depth=50, H=100, W=100)
    >>> volume = np.random.rand(50, 100, 100)
    >>> # Kernel: Small 3D cube (3x3x3)
    >>> kernel_vol = np.ones((3, 3, 3))
    >>> # channels_first=True tells function Depth is dim 0
    >>> # pad_channels=True allows sliding vertically through the volume (Depth)
    >>> out = correlate(volume, kernel_vol, channels_first=True, pad_channels=True)
    >>> # Result: (50, 100, 100) - Output is still a 3D volume
    """

    disp_correlate.detect_backend(img, backend)
    img, kernel = disp_correlate.cast_values(img, kernel)

    return disp_correlate(
        img,
        kernel,
        channels_first,
        stride,
        stride_channels,
        padding,
        pad_channels,
    )


def convolve(
    img: Any,
    kernel: Any | Tuple[Any, Any],
    channels_first: bool = False,
    stride: int | Tuple[int, int] = 1,
    stride_channels: bool = False,
    padding: Optional[str | float | int] = "same",
    pad_channels: bool = False,
    backend: Optional[str] = None,
) -> Any:
    """
    Performs a generic convolution on 2D images (spatial) or 3D volumes (volumetric).

    This function unifies standard 2D convolution, CNN-style multi-channel convolution,
    and true 3D volumetric convolution into a single interface. It is mathematically
    equivalent to flipping the `kernel` and performing a correlation.

    The behavior is determined by the shape of the input `kernel` and the `pad_channels` flag.

    Parameters:
    -----------
    img : np.ndarray
        Input data. Can be:
        - 2D: (H, W) or (C, H, W) / (H, W, C)
        - 3D: (D, H, W)
        Note: This function expects a single sample, not a batch of images.
    kernel : np.ndarray | Tuple[np.ndarray, ...]
        The filter to convolve with. The function automatically flips this kernel
        before application.
        - For 2D/CNN: shape (kH, kW) or (kH, kW, kCh).
        - For 3D: shape (kD, kH, kW).
        - Separable: A tuple of 1D arrays `(k_vert, k_horz)` or `(k_depth, k_vert, k_horz)`.
    channels_first : bool (default=False)
        If True, the first dimension is treated as the channel/depth dimension.
    stride : int | Tuple[int, int] (default=1)
        Spatial stride (step size).
    stride_channels : bool (default=False)
        If True, the `stride` is also applied to the channel/depth dimension.
    padding : str | float | int (default='same')
        'same', 'valid', or a constant value (float/int).
    pad_channels : bool (default=False)
        **CRITICAL PARAMETER**: Controls the "mode" of operation and kernel flipping.
        - `False` (CNN Mode): The channel dimension is NOT padded. The kernel is flipped
          along spatial axes (H, W) only; the channel order is preserved. Use this
          for standard Deep Learning convolutions.
        - `True` (Volumetric Mode): The channel dimension IS padded. The kernel is flipped
          along all axes (D, H, W). Use this for true 3D signal processing.
    backend : str, optional (default=None)
        Force a specific backend.

    Returns:
    --------
    output : Any
        The convolved image.

    Examples:
    ---------
    >>> import numpy as np

    **1. Standard 2D Convolution (Grayscale)**
    >>> img = np.random.rand(100, 100)
    >>> kernel = np.ones((3, 3))
    >>> # Kernel is spatially flipped before application
    >>> out = convolution(img, kernel)
    >>> # Result: (100, 100) - Standard spatial filtering

    **2. CNN-Style Convolution (RGB Input)**
    >>> # Input: (H=100, W=100, C=3)
    >>> img_rgb = np.random.rand(100, 100, 3)
    >>> # Kernel: (H=3, W=3, C=3) -> Matches input depth
    >>> kernel_cnn = np.ones((3, 3, 3))
    >>> # pad_channels=False ensures we don't pad depth (CNN behavior)
    >>> # Kernel is flipped on H and W axes, but NOT on C axis
    >>> out = convolution(img_rgb, kernel_cnn, pad_channels=False)
    >>> # Result: (100, 100, 1)

    **3. Volumetric Convolution (3D Medical Scan)**
    >>> # Input: (Depth=50, H=100, W=100)
    >>> volume = np.random.rand(50, 100, 100)
    >>> # Kernel: Small 3D cube (3x3x3)
    >>> kernel_vol = np.ones((3, 3, 3))
    >>> # pad_channels=True allows sliding vertically through the volume
    >>> # Kernel is flipped on ALL axes (Depth, Height, Width)
    >>> out = convolution(volume, kernel_vol, channels_first=True, pad_channels=True)
    >>> # Result: (50, 100, 100) - Output is still a 3D volume
    """

    disp_convolve.detect_backend(img, backend)
    img, kernel = disp_convolve.cast_values(img, kernel)

    return disp_convolve(
        img,
        kernel,
        channels_first,
        stride,
        stride_channels,
        padding,
        pad_channels,
    )


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

    disp_img2patches.detect_backend(img, backend)
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
