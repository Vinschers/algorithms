from vicentin.utils import array, isnan, sum, zeros

from vicentin.image.utils import convolve, gaussian_filter
from vicentin.image.differentiation import grad


def horn_schunck(img1, img2, u0, v0, alpha=100, iters=100, blur=1):
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

    H, W = img2.shape[:2]

    img1 = gaussian_filter(img1, blur)
    img2 = gaussian_filter(img2, blur)

    # Set initial value for the flow vectors
    u = u0.copy()
    v = v0.copy()

    # Estimate spatiotemporal derivatives
    fx, fy = grad(img1)
    ft = img2 - img1

    avg_kernel = array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]) / 12

    for _ in range(iters):
        # Compute local averages of the flow vectors using kernel_1
        uAvg = convolve(u, avg_kernel, mode="same")
        vAvg = convolve(v, avg_kernel, mode="same")

        # Compute flow vectors constrained by its local average and the optical flow constraints
        aux = (uAvg * fx + vAvg * fy + ft) / (alpha**2 + sum(fx**2 + fy**2))

        u = uAvg - fx * aux
        v = vAvg - fy * aux

    u[isnan(u)] = 0
    v[isnan(v)] = 0

    mvf = zeros((H, W, 2))
    mvf[..., 0] = v
    mvf[..., 1] = u

    return mvf
