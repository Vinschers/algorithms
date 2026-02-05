import pytest
import numpy as np

from vicentin.image.utils import correlate, convolve

BACKENDS = ["numpy", "torch"]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------


def generate_data(shape):
    """Generates random float data for a given shape."""
    return np.random.rand(*shape).astype(np.float32)


def assert_arrays_equal(arr1, arr2, tol=1e-5):
    """Helper to check array equality with tolerance."""
    np.testing.assert_allclose(arr1, arr2, rtol=tol, atol=tol)


def flip_kernel(kernel, pad_channels):
    """
    Mimics the flipping logic described in the docstring:
    - If pad_channels=False (CNN): Flip spatial dims only.
    - If pad_channels=True (Volumetric): Flip all dims.
    """
    kernel = np.array(kernel)
    if pad_channels:
        # Flip all dimensions (D, H, W)
        return np.flip(kernel)
    else:
        # CNN Mode: We assume the last dimension is Channel (H, W, C)
        # or first is Channel (C, H, W).
        if kernel.ndim == 2:
            return np.flip(kernel)
        elif kernel.ndim == 3:
            # Flip first two dims (H, W), leave C alone
            return kernel[::-1, ::-1, :]
        return kernel


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
class TestBasicFunctionality:
    """Basic sanity checks for shapes and 2D/3D operations."""

    def test_2d_grayscale_identity(self, backend):
        """Simple 2D correlation with an identity kernel should return input."""
        img = np.random.rand(10, 10)
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])  # Identity

        res = correlate(img, kernel, padding="same", backend=backend)
        assert_arrays_equal(res, img)

    def test_cnn_mode_output_shape(self, backend):
        """
        Docstring Example 2:
        Input (H, W, C), Kernel (H, W, C), pad_channels=False.
        Output should be (H, W) -> flattened 2D feature map.
        """
        img = generate_data((20, 20, 3))
        kernel = generate_data((3, 3, 3))

        # Explicitly pass padding="same" to ensure consistent shape (20, 20)
        out = correlate(
            img, kernel, pad_channels=False, padding="same", backend=backend
        )
        assert out.shape == (20, 20, 1)

    def test_volumetric_mode_output_shape(self, backend):
        """
        Docstring Example 3:
        Input (D, H, W), Kernel (D, H, W), pad_channels=True, channels_first=True.
        Output should be (D, H, W) -> preserves volume.
        """
        # (Depth, Height, Width)
        img = generate_data((10, 20, 20))
        kernel = generate_data((3, 3, 3))

        # Explicitly pass padding="same" to ensure consistent shape (10, 20, 20)
        out = correlate(
            img,
            kernel,
            channels_first=True,
            pad_channels=True,
            padding="same",
            backend=backend,
        )
        assert out.shape == (10, 20, 20)


class TestBackendConsistency:
    """Ensures all backends produce identical results."""

    @pytest.mark.skipif(
        len(BACKENDS) < 2, reason="Need at least 2 backends to compare"
    )
    @pytest.mark.parametrize("pad_channels", [True, False])
    @pytest.mark.parametrize("padding", ["same", None, "reflect", "wrap"])
    def test_backends_match(self, pad_channels, padding):
        """Checks consistency across all supported padding modes."""
        img = generate_data((15, 15, 3))
        kernel = generate_data((3, 3, 3))

        results = []
        for backend in BACKENDS:
            res = correlate(
                img,
                kernel,
                pad_channels=pad_channels,
                padding=padding,
                backend=backend,
            )
            results.append(res)

        base_res = results[0]
        for res in results[1:]:
            assert_arrays_equal(base_res, res)


class TestMathInvariants:
    """Checks mathematical properties (Convolution vs Correlation)."""

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("pad_channels", [True, False])
    def test_convolve_equals_flipped_correlate(self, backend, pad_channels):
        """Verify: convolve(img, K) == correlate(img, flip(K))"""
        img = generate_data((12, 12, 3))
        kernel = generate_data((3, 3, 3))

        conv_res = convolve(
            img, kernel, pad_channels=pad_channels, backend=backend
        )
        flipped_kernel = flip_kernel(kernel, pad_channels=pad_channels)
        corr_res = correlate(
            img, flipped_kernel, pad_channels=pad_channels, backend=backend
        )

        assert_arrays_equal(conv_res, corr_res)


class TestPaddingAndStrides:
    """Comprehensive check of numerical padding and stride combinations."""

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("stride", [1, 2, (2, 2)])
    @pytest.mark.parametrize("padding", ["same", None, 0.0, 1.5])
    def test_2d_strides_and_padding(self, backend, stride, padding):
        img = generate_data((20, 20))
        kernel = generate_data((3, 3))

        try:
            out = correlate(
                img, kernel, stride=stride, padding=padding, backend=backend
            )

            if stride == 2 or stride == (2, 2):
                if padding == "same":
                    assert out.shape == (10, 10)
                elif padding is None:
                    assert out.shape == (9, 9)

        except Exception as e:
            pytest.fail(f"Failed with stride={stride}, padding={padding}: {e}")

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_channel_stride(self, backend):
        """Test stride_channels=True logic."""
        img = generate_data((10, 10, 10))
        kernel = generate_data((3, 3, 3))

        out = correlate(
            img,
            kernel,
            pad_channels=True,
            stride=2,
            stride_channels=True,
            padding="same",
            backend=backend,
        )
        assert out.shape == (5, 5, 5)


class TestBoundaryModes:
    """Tests for complex boundary conditions (Reflect, Wrap, Extend)."""

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("mode", ["reflect", "wrap", "extend"])
    def test_boundary_modes_run(self, backend, mode):
        """Ensure boundary modes execute and preserve shape (like 'same')."""
        img = generate_data((20, 20))
        kernel = generate_data((3, 3))

        # These modes should behave like 'same' regarding output shape
        out = correlate(img, kernel, padding=mode, backend=backend)
        assert out.shape == (20, 20)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_wrap_behavior(self, backend):
        """Specific check for 'wrap' logic."""
        # [10, 0, ..., 0]
        # Wrapping ensures the 10 wraps around to the other side.
        img = np.zeros((10, 10))
        img[0, 5] = 100.0  # Top edge

        # Kernel that looks 'up' (above current pixel)
        # 3x3 kernel with 1 at bottom center: [[0,0,0], [0,0,0], [0,1,0]]
        # When centered at (9, 5) (bottom edge), it looks at (9+1, 5) -> (0, 5) which is 100.
        kernel = np.zeros((3, 3))
        kernel[2, 1] = 1.0

        # We need 'wrap' padding for the bottom pixel to see the top pixel
        out = correlate(img, kernel, padding="wrap", backend=backend)

        # The pixel at bottom edge (9, 5) should 'see' the top edge (0, 5)
        # Note: tolerance is high because of float precision diffs in backends
        assert np.isclose(out[9, 5], 100.0, atol=1e-4)


class TestSeparableKernels:
    """Test the Tuple[Any, Any] kernel input support."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_separable_2d(self, backend):
        img = generate_data((20, 20))
        k_y = generate_data((3,))
        k_x = generate_data((3,))

        out_sep = correlate(img, (k_y, k_x), backend=backend)
        k_full = np.outer(k_y, k_x)
        out_full = correlate(img, k_full, backend=backend)

        assert_arrays_equal(out_sep, out_full)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_separable_3d(self, backend):
        img = generate_data((10, 20, 20))
        k_d = generate_data((3,))
        k_h = generate_data((3,))
        k_w = generate_data((3,))

        out_sep = correlate(
            img,
            (k_d, k_h, k_w),
            channels_first=True,
            pad_channels=True,
            backend=backend,
        )
        assert out_sep.shape == (10, 20, 20)


class TestEdgeCases:
    """Tests for unusual but valid inputs."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_kernel_larger_than_image(self, backend):
        img = generate_data((5, 5))
        kernel = generate_data((7, 7))
        out = correlate(img, kernel, padding="same", backend=backend)
        assert out.shape == (5, 5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_1x1_convolution(self, backend):
        img = np.ones((10, 10)) * 2.0
        kernel = np.ones((1, 1)) * 3.0
        out = correlate(img, kernel, backend=backend)
        assert np.allclose(out, 6.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_asymmetric_kernel(self, backend):
        img = generate_data((10, 10))
        kernel = generate_data((1, 5))
        out = correlate(img, kernel, padding="same", backend=backend)
        assert out.shape == (10, 10)
