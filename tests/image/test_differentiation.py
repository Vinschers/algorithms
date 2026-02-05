import pytest
import numpy as np
import torch
from typing import Tuple

from vicentin.image.differentiation import finite_diffs, sobel, grad, laplacian

# ==========================================
# Helpers & Fixtures
# ==========================================


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float()


def _get_ramp_image(shape: Tuple[int, ...], axis: int = 0) -> np.ndarray:
    grids = np.indices(shape)
    return grids[axis].astype(np.float32)


def _get_parabola_image(shape: Tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape)
    return (x**2 + y**2).astype(np.float32)


BACKENDS = ["numpy", "torch"]
SHAPES_2D = [(10, 10), (32, 32)]
BOUNDARIES = ["reflect", "wrap", "extend"]

# ==========================================
# 1. Finite Differences Tests
# ==========================================


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("mode", ["central", "forward", "backward"])
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_finite_diffs_shapes_and_types(backend, shape, mode, boundary):
    img_np = np.random.rand(*shape).astype(np.float32)
    img = _to_torch(img_np) if backend == "torch" else img_np

    dx, dy = finite_diffs(img, mode=mode, boundary=boundary, backend=backend)

    if backend == "numpy":
        assert isinstance(dx, np.ndarray)
        assert isinstance(dy, np.ndarray)
    else:
        assert isinstance(dx, torch.Tensor)
        assert isinstance(dy, torch.Tensor)

    assert dx.shape == shape
    assert dy.shape == shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_finite_diffs_correctness_ramp(backend):
    img_np = _get_ramp_image((10, 10), axis=1)
    img = _to_torch(img_np) if backend == "torch" else img_np

    dx, dy = finite_diffs(
        img, mode="central", boundary="extend", backend=backend
    )

    if backend == "torch":
        dx, dy = dx.numpy(), dy.numpy()

    valid_dx = dx[1:-1, 1:-1]
    valid_dy = dy[1:-1, 1:-1]

    assert np.allclose(valid_dx, 1.0, atol=1e-5)
    assert np.allclose(valid_dy, 0.0, atol=1e-5)


@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.parametrize("backend", BACKENDS)
def test_finite_diffs_3d_channels(channels_first, backend):
    shape = (3, 10, 10) if channels_first else (10, 10, 3)
    img_np = np.random.rand(*shape).astype(np.float32)
    img = _to_torch(img_np) if backend == "torch" else img_np

    dx, dy = finite_diffs(
        img, channels_first=channels_first, boundary="reflect", backend=backend
    )

    if backend == "torch":
        dx, dy = dx.numpy(), dy.numpy()

    assert dx.shape == shape
    assert dy.shape == shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_valid_boundary_clipping(backend):
    img_np = np.zeros((10, 10), dtype=np.float32)
    img = _to_torch(img_np) if backend == "torch" else img_np

    dx, dy = finite_diffs(img, mode="central", boundary=None, backend=backend)

    expected_shape = (8, 8)
    if backend == "torch":
        assert tuple(dx.shape) == expected_shape
    else:
        assert dx.shape == expected_shape


# ==========================================
# 2. Sobel Tests
# ==========================================


@pytest.mark.parametrize("backend", BACKENDS)
def test_sobel_correctness_impulse(backend):
    """
    Verifies Sobel kernel application using an impulse image.
    The implementation uses correlation with [-1, 0, 1].
    The impulse response of correlation(I, K) is Flipped(K).
    Therefore, the expected output should be [1, 0, -1].
    """
    img_np = np.zeros((5, 5), dtype=np.float32)
    img_np[2, 2] = 1.0
    img = _to_torch(img_np) if backend == "torch" else img_np

    dx, dy = sobel(img, mode="forward", boundary="pad", backend=backend)

    if backend == "torch":
        dx = dx.numpy()

    # Kernel constructed in sobel is [-1, 0, 1]
    # Correlation with [-1, 0, 1] yields gradient Right - Left.
    # Impulse response (flipped kernel) is [1, 0, -1].
    # [1, 2, 1]^T * [1, 0, -1]
    expected_impulse_x = np.array(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32
    )

    center_dx = dx[1:4, 1:4]
    assert np.allclose(center_dx, expected_impulse_x, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sobel_orientation(backend):
    # Vertical Line Image
    img_v = np.zeros((10, 10), dtype=np.float32)
    img_v[:, 5] = 1.0

    img = _to_torch(img_v) if backend == "torch" else img_v

    dx, dy = sobel(img, boundary="pad", backend=backend)

    if backend == "torch":
        dx, dy = dx.numpy(), dy.numpy()

    assert np.abs(dx).max() > 0.5
    assert np.allclose(dy[1:-1, :], 0.0, atol=1e-5)


# ==========================================
# 3. Grad Wrapper Tests
# ==========================================


@pytest.mark.parametrize("method", ["diff", "sobel"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_grad_consistency(method, backend):
    img_np = np.random.rand(10, 10).astype(np.float32)
    img = _to_torch(img_np) if backend == "torch" else img_np

    dx, dy = grad(img, method=method, backend=backend)

    assert dx is not None
    assert dy is not None

    shape = (10, 10)
    if backend == "torch":
        assert tuple(dx.shape) == shape
    else:
        assert dx.shape == shape


# ==========================================
# 4. Laplacian Tests
# ==========================================


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["direct", "diff"])
def test_laplacian_parabola(backend, method):
    shape = (20, 20)
    img_np = _get_parabola_image(shape)
    img = _to_torch(img_np) if backend == "torch" else img_np

    lap = laplacian(img, method=method, boundary="extend", backend=backend)

    if backend == "torch":
        lap = lap.numpy()

    center_val = lap[5:-5, 5:-5]

    if method == "direct":
        expected_val = 12.0
    else:
        expected_val = 4.0

    assert np.allclose(
        center_val, expected_val, atol=1e-4
    ), f"Method {method} failed. Expected ~{expected_val}, got {center_val.mean()}"


# ==========================================
# 5. Error Handling
# ==========================================


def test_invalid_backend():
    img = np.zeros((10, 10))
    try:
        finite_diffs(img, backend="invalid_backend")
    except ValueError:
        pass
    except Exception as e:
        pytest.fail(f"Raised unexpected exception for invalid backend: {e}")
