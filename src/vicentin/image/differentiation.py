from vicentin.utils import array, flip, roll
from vicentin.image.utils import convolve


def finite_diffs(img, backward=False):
    """
    Compute the finite difference (discrete derivative) of an image along x and y axes.

    Parameters
    ----------
    img : numpy.ndarray or jax.numpy.ndarray
        Input image (or 2D array).
    backward : bool, optional
        If False (default), computes the forward difference.
        If True, computes the backward difference.
        The scaling factor -shift ensures that the returned difference matches the
        standard finite difference convention:
            forward: f(x+1) - f(x)
            backward: f(x) - f(x-1)

    Returns
    -------
    numpy.ndarray or jax.numpy.ndarray
        A 2-element array where:
          - element 0 is the difference along axis=1 (x-direction),
          - element 1 is the difference along axis=0 (y-direction).
    """
    shift = 1 if backward else -1

    return -shift * array([roll(img, shift, 1) - img, roll(img, shift, 0) - img])


def sobel(img, backward=False):
    """
    Compute the Sobel gradient of an image by convolving with Sobel kernels.

    Parameters
    ----------
    img : numpy.ndarray or jax.numpy.ndarray
        Input image.
    backward : bool, optional
        If False (default), uses the standard (forward) Sobel kernels.
        If True, flips the Sobel kernels to approximate the adjoint (backward) operator.

    Returns
    -------
    numpy.ndarray or jax.numpy.ndarray
        A 2-element array where:
          - element 0 is the result of convolving img with the x-direction Sobel kernel,
          - element 1 is the result of convolving img with the y-direction Sobel kernel.
    """
    sobel_x = array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    if backward:
        sobel_x, sobel_y = flip(sobel_x), flip(sobel_y)

    return array([convolve(img, sobel_x), convolve(img, sobel_y)])


def canny(img, backward=False):
    """
    Placeholder for Canny edge detection method.

    Parameters
    ----------
    img : numpy.ndarray or jax.numpy.ndarray
        Input image.
    backward : bool, optional
        If True, should compute the backward (adjoint) operation.
        (Not implemented.)

    Raises
    ------
    NotImplementedError
        Always raised as Canny edge detection is not implemented.
    """
    raise NotImplementedError("Canny edge detection is not implemented yet.")


def grad(img, method=None, backward=False):
    """
    Compute the gradient of an image using a specified method.

    Parameters
    ----------
    img : numpy.ndarray or jax.numpy.ndarray
        Input image.
    method : str, optional
        The gradient computation method. Options:
          - "diff" for finite differences (default if None).
          - "sobel" for Sobel operator.
          - "canny" for Canny edge detection (not implemented).
    backward : bool, optional
        If False (default), compute the forward gradient.
        If True, compute the backward (adjoint) gradient operator.

    Returns
    -------
    numpy.ndarray or jax.numpy.ndarray
        A 2-element array with the gradient components along x (index 0)
        and y (index 1).

    Raises
    ------
    ValueError
        If an unknown method is specified.
    """
    if method is None:
        method = "diff"

    if method == "diff":
        return finite_diffs(img, backward)
    elif method == "sobel":
        return sobel(img, backward)
    elif method == "canny":
        return canny(img, backward)
    else:
        raise ValueError(f"Unknown gradient method: {method}")


def laplacian(img, method=None):
    """
    Compute the Laplacian of an image using the divergence of the gradient.

    The Laplacian is computed as:
        Laplacian(I) = \\nabla^\\top \\nabla I
    where the transposed gradient is approximated by applying the backward operator.

    Parameters
    ----------
    img : numpy.ndarray or jax.numpy.ndarray
        Input image.
    method : str, optional
        The gradient method to use ("diff", "sobel", or "canny").
        Defaults to "diff" if None is provided.

    Returns
    -------
    numpy.ndarray or jax.numpy.ndarray
        The Laplacian of the image.
    """
    g = grad(img, method)

    lap_x = grad(g[0], method, backward=True)[0]
    lap_y = grad(g[1], method, backward=True)[1]

    return lap_x + lap_y
