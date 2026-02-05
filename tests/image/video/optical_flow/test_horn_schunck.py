import pytest
import numpy as np
import torch
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter

from vicentin.image.video.optical_flow import horn_schunck


@pytest.fixture
def texture_pattern():
    """Generates a 50x50 random texture with gradients."""

    def _generate(shift_x=0, shift_y=0):
        np.random.seed(42)
        base = np.random.rand(50, 50).astype(np.float32)
        # Sigma 1.0 preserves edges better than 2.0 -> stronger gradients
        base = scipy_gaussian_filter(base, sigma=1.0)
        # Normalize to [0, 1]
        base = (base - base.min()) / (base.max() - base.min())

        shifted = np.roll(base, shift_x, axis=1)
        shifted = np.roll(shifted, shift_y, axis=0)
        return shifted

    return _generate


@pytest.fixture
def random_noise():
    np.random.seed(1337)
    img1 = np.random.rand(40, 40).astype(np.float32)
    img2 = np.random.rand(40, 40).astype(np.float32)
    return img1, img2


def get_center_flow(mvf, window=10):
    h, w = mvf.shape[:2]
    cy, cx = h // 2, w // 2
    center_region = mvf[cy - window : cy + window, cx - window : cx + window, :]
    return np.mean(center_region, axis=(0, 1))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_no_motion(backend, texture_pattern):
    img1 = texture_pattern(0, 0)
    img2 = texture_pattern(0, 0)
    mvf = horn_schunck(img1, img2, alpha=0.1, iters=50, backend=backend)
    if backend == "torch":
        mvf = mvf.cpu().numpy()
    assert np.allclose(mvf, 0, atol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_simple_translation_right(backend, texture_pattern):
    img1 = texture_pattern(0, 0)
    img2 = texture_pattern(1, 0)  # Right 1px

    # Alpha=0.1 is crucial for [0, 1] float images
    mvf = horn_schunck(
        img1, img2, alpha=0.1, iters=200, blur=0.5, backend=backend
    )

    if backend == "torch":
        mvf = mvf.cpu().numpy()
    v_mean, u_mean = get_center_flow(mvf)

    # Now we expect robust flow closer to 1.0
    assert u_mean > 0.7, f"Expected u > 0.7, got {u_mean}"
    assert abs(v_mean) < 0.1, f"Expected v near 0, got {v_mean}"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_simple_translation_down(backend, texture_pattern):
    img1 = texture_pattern(0, 0)
    img2 = texture_pattern(0, 1)  # Down 1px

    mvf = horn_schunck(
        img1, img2, alpha=0.1, iters=200, blur=0.5, backend=backend
    )

    if backend == "torch":
        mvf = mvf.cpu().numpy()
    v_mean, u_mean = get_center_flow(mvf)

    assert v_mean > 0.7, f"Expected v > 0.7, got {v_mean}"
    assert abs(u_mean) < 0.1, f"Expected u near 0, got {u_mean}"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_initialization_pass_through(backend, texture_pattern):
    img1 = texture_pattern(0, 0)
    img2 = texture_pattern(0, 0)
    shape = img1.shape
    u0 = np.ones(shape, dtype=np.float32) * 5.0
    v0 = np.ones(shape, dtype=np.float32) * 5.0
    if backend == "torch":
        u0 = torch.from_numpy(u0)
        v0 = torch.from_numpy(v0)

    mvf = horn_schunck(
        img1, img2, u0=u0, v0=v0, alpha=100.0, iters=5, backend=backend
    )
    if backend == "torch":
        mvf = mvf.cpu().numpy()

    assert np.mean(mvf) > 4.5
