import pytest
import numpy as np
import torch

from vicentin.image.differentiation import finite_diffs, sobel, grad, laplacian


class ImageFactory:
    @staticmethod
    def get_linear_ramp(backend, shape=(10, 10), slopes=(1.0, 2.0)):
        """
        Creates an image I(y, x) = slopes[0]*x + slopes[1]*y.
        Constant gradient: dx = slopes[0], dy = slopes[1].
        """
        y, x = np.mgrid[: shape[0], : shape[1]]
        img_np = slopes[0] * x + slopes[1] * y
        img_np = img_np.astype(np.float32)

        if backend == "numpy":
            return img_np
        elif backend == "pytorch":
            return torch.tensor(img_np, dtype=torch.float32)

    @staticmethod
    def get_parabola(backend, shape=(10, 10)):
        """
        Creates an image I(y, x) = x^2 + y^2.
        Analytical Laplacian: 2 + 2 = 4.
        """
        y, x = np.mgrid[: shape[0], : shape[1]]
        img_np = x**2 + y**2
        img_np = img_np.astype(np.float32)

        if backend == "numpy":
            return img_np
        elif backend == "pytorch":
            return torch.tensor(img_np, dtype=torch.float32)

    @staticmethod
    def get_impulse(backend, shape=(5, 5)):
        """
        Creates an image with a single 1.0 in the center, 0 elsewhere.
        Useful for kernel inspection.
        """
        img_np = np.zeros(shape, dtype=np.float32)
        img_np[shape[0] // 2, shape[1] // 2] = 1.0

        if backend == "numpy":
            return img_np
        elif backend == "pytorch":
            return torch.tensor(img_np, dtype=torch.float32)

    @staticmethod
    def to_numpy(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        return val

    @staticmethod
    def assert_close(val1, val2, backend, atol=1e-5):
        val1_np = ImageFactory.to_numpy(val1)
        val2_np = ImageFactory.to_numpy(val2)

        # Handle tuple returns (dx, dy)
        if isinstance(val1_np, tuple):
            for v1, v2 in zip(val1_np, val2_np):
                np.testing.assert_allclose(v1, v2, atol=atol)
        else:
            np.testing.assert_allclose(val1_np, val2_np, atol=atol)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_finite_diffs_central_linear(backend):
    """
    Test 1: Exact reconstruction of constant gradient using central differences.
    Input: z = 1.0*x + 2.0*y
    Expected: dx=1.0, dy=2.0 (everywhere, except borders depending on pad)
    """
    img = ImageFactory.get_linear_ramp(backend, slopes=(1.0, 2.0))
    dx, dy = finite_diffs(img, mode="central", backend=backend)

    # Central difference is exact for linear functions
    # Check inner region to avoid boundary artifacts
    inner_dx = dx[1:-1, 1:-1]
    inner_dy = dy[1:-1, 1:-1]

    ImageFactory.assert_close(inner_dx, 1.0, backend)
    ImageFactory.assert_close(inner_dy, 2.0, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_finite_diffs_forward_backward(backend):
    """
    Test 2: Check consistency between forward and backward modes.
    Forward diff at x should equal Backward diff at x+1 for linear functions.
    """
    img = ImageFactory.get_linear_ramp(backend)

    dx_fwd, _ = finite_diffs(img, mode="forward", backend=backend)
    dx_bwd, _ = finite_diffs(img, mode="backward", backend=backend)

    # Shift backward result to align with forward
    # fwd[i] = x[i+1] - x[i]
    # bwd[i+1] = x[i+1] - x[i]
    if backend == "numpy":
        match = np.allclose(dx_fwd[1:-2, 1:-2], dx_bwd[1:-2, 2:-1], atol=1e-5)
    else:
        match = torch.allclose(
            dx_fwd[1:-2, 1:-2], dx_bwd[1:-2, 2:-1], atol=1e-5
        )

    assert match


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_sobel_impulse_response(backend):
    """
    Test 3: Apply Sobel to an impulse to verify kernel values.
    Impulse at center.
    Sobel X kernel central row should be [-2, 0, 2].
    """
    img = ImageFactory.get_impulse(backend, shape=(5, 5))
    dx, dy = sobel(img, mode="central", backend=backend)

    res_dx = ImageFactory.to_numpy(dx)

    # Check the immediate neighbors of the center (2,2)
    assert np.isclose(res_dx[2, 1], 2.0)
    assert np.isclose(res_dx[2, 3], -2.0)
    # Center should be 0
    assert np.isclose(res_dx[2, 2], 0.0)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_grad_dispatcher(backend):
    """
    Test 4: Verify `grad` calls the correct underlying method.
    """
    img = ImageFactory.get_linear_ramp(backend)

    # Method "diff" should match finite_diffs exactly
    g_dx, g_dy = grad(img, method="diff", mode="central", backend=backend)
    f_dx, f_dy = finite_diffs(img, mode="central", backend=backend)

    ImageFactory.assert_close(g_dx, f_dx, backend)

    # Method "sobel" should match sobel exactly
    g_dx_s, g_dy_s = grad(img, method="sobel", backend=backend)
    s_dx, s_dy = sobel(img, backend=backend)

    ImageFactory.assert_close(g_dx_s, s_dx, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_laplacian_parabola_direct(backend):
    """
    Test 5: Laplacian of a parabola z = x^2 + y^2.
    Analytical result is constant 4.0.
    Method: 'direct' (convolution)
    """
    img = ImageFactory.get_parabola(backend)
    lap = laplacian(img, method="direct", backend=backend)

    # Ignore boundaries
    inner = lap[1:-1, 1:-1]
    # The standard 3x3 laplacian kernel scaled properly typically sums to 0,
    # but for x^2, the finite diff (1, -2, 1) yields exactly 2.
    # dx2 = (x+1)^2 - 2x^2 + (x-1)^2 = x^2 + 2x + 1 - 2x^2 + x^2 - 2x + 1 = 2
    # dy2 = 2
    # sum = 4
    ImageFactory.assert_close(inner, 4.0, backend, atol=1e-4)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_laplacian_parabola_iterative(backend):
    """
    Test 6: Laplacian of a parabola via iterative gradients.
    Method: 'diff'
    """
    img = ImageFactory.get_parabola(backend)
    lap = laplacian(img, method="diff", backend=backend)

    inner = lap[1:-1, 1:-1]
    ImageFactory.assert_close(inner, 4.0, backend, atol=1e-4)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_batch_dimensions_4d(backend):
    """
    Test 7: Ensure 4D inputs (B, C, H, W) are handled and shape is preserved.
    """
    shape = (2, 3, 10, 10)  # B=2, C=3
    if backend == "numpy":
        img = np.random.rand(*shape).astype(np.float32)
    else:
        img = torch.randn(*shape)

    dx, dy = finite_diffs(img, backend=backend)

    assert dx.shape == shape
    assert dy.shape == shape


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_batch_dimensions_2d(backend):
    """
    Test 8: Ensure 2D inputs (H, W) are handled without adding dims to output.
    """
    shape = (10, 10)
    img = ImageFactory.get_linear_ramp(backend, shape)

    dx, dy = finite_diffs(img, backend=backend)

    assert dx.shape == shape
    assert dy.shape == shape


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_boundary_reflect(backend):
    """
    Test 9: Boundary condition 'reflect'.
    If we have a ramp 1, 2, 3. Reflected padding makes it 2, 1, 2, 3, 2.
    Gradient at left edge (forward): 2-1 = 1.
    """
    # Create 1D horizontal ramp as image
    row = np.arange(10, dtype=np.float32)
    img_np = np.tile(row, (10, 1))  # (10, 10)

    if backend == "numpy":
        img = img_np
    else:
        img = torch.tensor(img_np)

    # Using 'forward' diff with 'reflect'
    # At index 0: val is 0. Pad left is 1 (index 1).
    # Wait, reflect padding of [0, 1, 2] is [1, 0, 1, 2].
    # But usually finite_diffs pads (1,1).
    dx, _ = finite_diffs(
        img, mode="forward", boundary="reflect", backend=backend
    )

    # We just check it doesn't crash and returns valid shape for now,
    # exact boundary math depends on specific reflect implementation (dcb|abcd|cba vs cb|abcd|cb)
    assert dx.shape == (10, 10)

    # Check interior is constant 1
    ImageFactory.assert_close(dx[:, 1:-1], 1.0, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_constant_image_zero_grad(backend):
    """
    Test 10: A constant image should have 0 gradient everywhere.
    """
    if backend == "numpy":
        img = np.ones((10, 10), dtype=np.float32) * 5.0
    else:
        img = torch.ones((10, 10), dtype=torch.float32) * 5.0

    dx, dy = grad(img, method="sobel", backend=backend)

    ImageFactory.assert_close(dx, 0.0, backend)
    ImageFactory.assert_close(dy, 0.0, backend)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_dtype_casting(backend):
    """
    Test 11: Pass integer input, ensure output is float.
    """
    shape = (10, 10)
    if backend == "numpy":
        img = np.arange(100).reshape(shape).astype(np.int32)
    else:
        img = torch.arange(100).reshape(shape).to(torch.int32)

    dx, dy = finite_diffs(img, backend=backend)

    if backend == "numpy":
        assert dx.dtype == np.float32
    else:
        assert dx.dtype == torch.float32


def test_invalid_mode_raises():
    """
    Test 12: Invalid mode strings should raise ValueError.
    (Backend agnostic check, running on numpy is sufficient)
    """
    img = np.zeros((10, 10))
    with pytest.raises(ValueError):
        finite_diffs(img, mode="invalid_mode", backend="numpy")

    with pytest.raises(ValueError):
        grad(img, method="magic_wand", backend="numpy")


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_laplacian_method_consistency(backend):
    """
    Test 13: Ensure 'direct' and 'diff' methods produce reasonably similar results
    on a smooth image (e.g. Gaussian), ignoring boundaries.
    """
    # Create simple Gaussian blob
    y, x = np.mgrid[-2:2:20j, -2:2:20j]
    img_np = np.exp(-(x**2 + y**2)).astype(np.float32)

    if backend == "numpy":
        img = img_np
    else:
        img = torch.tensor(img_np)

    lap_direct = laplacian(img, method="direct", backend=backend)
    lap_diff = laplacian(img, method="diff", backend=backend)

    # They won't be identical due to discretization diffs (3x3 kernel vs 2-step diffs),
    # but they should be correlated.
    # For this test, we accept a higher tolerance or check correlation.
    # Here we check they are close enough in L2 norm sense relative to signal energy
    diff = ImageFactory.to_numpy(lap_direct - lap_diff)
    error = np.linalg.norm(diff) / np.linalg.norm(
        ImageFactory.to_numpy(lap_direct)
    )

    assert error < 0.2  # <20% relative difference expected between kernels


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_sobel_backward_mode(backend):
    """
    Test 14: Test Sobel 'backward' mode runs.
    """
    img = ImageFactory.get_linear_ramp(backend)
    # Just ensure it runs and produces gradients
    dx, dy = sobel(img, mode="backward", backend=backend)

    ImageFactory.assert_close(dx[1:-1, 1:-1], -8.0, backend, atol=0.1)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_auto_backend_detection(backend):
    """
    Test 15: Passing a Tensor without specifying backend="pytorch" should auto-detect.
    """
    img = torch.zeros((10, 10))
    dx, dy = finite_diffs(img)  # backend=None

    assert isinstance(dx, torch.Tensor)
    assert isinstance(dy, torch.Tensor)

    img_np = np.zeros((10, 10))
    dx_np, dy_np = finite_diffs(img_np)  # backend=None

    assert isinstance(dx_np, np.ndarray)
