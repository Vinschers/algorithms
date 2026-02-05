import pytest
import numpy as np

from vicentin.image.utils import (
    pad_image,
    gaussian_filter,
    img2patches,
    get_neighbors,
)

BACKENDS = ["numpy", "torch"]


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------


def generate_data(shape):
    return np.random.rand(*shape).astype(np.float32)


def assert_arrays_equal(arr1, arr2, tol=1e-5):
    np.testing.assert_allclose(arr1, arr2, rtol=tol, atol=tol)


# -----------------------------------------------------------------------------
# TEST PAD IMAGE
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
class TestPadImage:
    def test_pad_2d_simple(self, backend):
        img = generate_data((10, 10))
        padding = (1, 1, 2, 2)
        padded = pad_image(img, padding, mode="constant", backend=backend)
        assert padded.shape == (12, 14)

    def test_pad_channels_last(self, backend):
        img = generate_data((10, 10, 3))
        padding = (1, 1, 1, 1)
        padded = pad_image(img, padding, mode="constant", backend=backend)
        assert padded.shape == (12, 12, 3)

    def test_pad_channels_first(self, backend):
        img = generate_data((3, 10, 10))
        padding = (1, 1, 1, 1)
        padded = pad_image(
            img, padding, mode="constant", channels_first=True, backend=backend
        )
        assert padded.shape == (3, 12, 12)

    def test_pad_volumetric_3d(self, backend):
        img = generate_data((10, 20, 20))
        padding = (1, 1, 2, 2, 0, 0)
        padded = pad_image(img, padding, mode="constant", backend=backend)
        assert padded.shape == (12, 24, 20)

    @pytest.mark.parametrize("mode", ["reflect", "wrap", "extend"])
    def test_pad_modes(self, backend, mode):
        # This 2D input previously failed on Torch without the unsqueeze fix
        img = generate_data((10, 10))
        padding = (2, 2, 2, 2)
        padded = pad_image(img, padding, mode=mode, backend=backend)
        assert padded.shape == (14, 14)

    def test_pad_constant_value(self, backend):
        img = np.zeros((5, 5))
        val = 5.5
        padded = pad_image(img, (1, 1, 1, 1), mode=val, backend=backend)
        assert np.allclose(padded[0, 0], val)


@pytest.mark.parametrize("backend", BACKENDS)
class TestGaussianFilter:
    def test_gaussian_shape_preservation(self, backend):
        img = generate_data((20, 20))
        out = gaussian_filter(img, sigma=1.0, backend=backend)
        assert out.shape == img.shape

    def test_gaussian_smoothing_effect(self, backend):
        img = np.random.rand(50, 50).astype(np.float32)
        blurred = gaussian_filter(img, sigma=2.0, backend=backend)

        # FIX: Cast to numpy for stats calculation to avoid Torch TypeError
        if backend == "torch":
            blurred = blurred.cpu().numpy()

        assert np.var(blurred) < np.var(img)

    def test_gaussian_flat_image(self, backend):
        # FIX: Use larger image and smaller sigma.
        # gaussian_filter uses zero-padding ('same').
        # If sigma is large (e.g. 5.0), the kernel is huge (radius~20).
        # Zeros from padding bleed into the center, lowering the value.
        img = np.ones((40, 40)) * 5.0
        out = gaussian_filter(img, sigma=0.5, backend=backend)

        if backend == "torch":
            out = out.cpu().numpy()

        center = out[10:-10, 10:-10]
        assert np.allclose(center, 5.0, atol=1e-3)

    def test_sigma_magnitude(self, backend):
        img = np.zeros((21, 21))
        img[10, 10] = 1.0

        out_small = gaussian_filter(img, sigma=0.5, backend=backend)
        out_large = gaussian_filter(img, sigma=2.0, backend=backend)

        if backend == "torch":
            out_small = out_small.cpu().numpy()
            out_large = out_large.cpu().numpy()

        assert out_large[10, 10] < out_small[10, 10]


@pytest.mark.parametrize("backend", BACKENDS)
class TestImg2Patches:
    # ... (Keep existing tests) ...
    def test_patch_shapes_perfect_fit(self, backend):
        img = generate_data((20, 20))
        patches = img2patches(img, (5, 5), 5, 5, backend=backend)
        assert patches.shape == (4, 4, 5, 5)

    def test_patch_content(self, backend):
        img = np.arange(16).reshape(4, 4).astype(np.float32)
        patches = img2patches(img, (2, 2), 2, 2, backend=backend)
        if backend == "torch":
            patches = patches.numpy()
        assert_arrays_equal(patches[0, 0], [[0, 1], [4, 5]])

    def test_overlapping_patches(self, backend):
        img = generate_data((10, 10))
        patches = img2patches(img, (5, 5), 1, 1, backend=backend)
        assert patches.shape == (6, 6, 5, 5)


@pytest.mark.parametrize("backend", BACKENDS)
class TestGetNeighbors:
    # ... (Keep existing 2D tests) ...
    def test_neighbors_2d_center(self, backend):
        img = np.zeros((10, 10))
        nbs = get_neighbors(img, 5, 5, neighborhood=4, backend=backend)
        assert len(nbs) == 4

    def test_neighbors_2d_corner(self, backend):
        img = np.zeros((10, 10))
        nbs = get_neighbors(img, 0, 0, neighborhood=4, backend=backend)
        assert len(nbs) == 2

    def test_neighbors_3d_volumetric(self, backend):
        """Voxel in 3D volume (H, W, D)."""
        # FIX: The implementation assumes (H, W, D) for neighbor checking.
        # We pass (10, 10, 5) so H=10, W=10, D=5.
        img = np.zeros((10, 10, 5))

        # Center at H=5, W=5, D=2
        nbs = get_neighbors(img, 5, 5, depth=2, neighborhood=4, backend=backend)

        # Should have 6 neighbors (Up, Down, Left, Right, Front, Back)
        assert len(nbs) == 6

        z_coords = [n[2] for n in nbs]
        assert 1 in z_coords
        assert 3 in z_coords

    def test_neighbors_8_connectivity(self, backend):
        img = np.zeros((10, 10))
        nbs = get_neighbors(img, 5, 5, neighborhood=8, backend=backend)
        assert len(nbs) == 8
