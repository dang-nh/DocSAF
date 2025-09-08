"""Unit tests for attenuation field module."""

import pytest
import torch
import numpy as np

from src.docsaf.field import (
    apply_field,
    apply_field_safe,
    field_with_smoothing,
    compute_tv_loss,
    validate_field_params,
    field_stats,
)


class TestFieldApplication:
    """Tests for field application functions."""

    def test_apply_field_basic(self):
        """Test basic field application."""
        # Create test inputs
        B, C, H, W = 2, 3, 32, 32
        device = "cpu"

        x = torch.rand(B, C, H, W, device=device)
        S = torch.rand(B, 1, H, W, device=device)
        alpha = 1.0
        radius = 5.0

        # Apply field
        x_adv = apply_field(x, S, alpha, radius)

        # Check output properties
        assert x_adv.shape == x.shape
        assert x_adv.dtype == x.dtype
        assert x_adv.device == x.device
        assert torch.all(x_adv >= 0.0) and torch.all(x_adv <= 1.0)

    def test_field_identity_alpha_zero(self):
        """Test that alpha=0 produces identity transformation."""
        x = torch.rand(1, 3, 24, 24)
        S = torch.rand(1, 1, 24, 24)
        alpha = 0.0
        radius = 5.0

        x_adv = apply_field(x, S, alpha, radius)

        # Should be nearly identical (within numerical precision)
        assert torch.allclose(x_adv, x, atol=1e-6)

    def test_field_monotonic_lpips(self):
        """Test that LPIPS increases monotonically with alpha."""
        try:
            import lpips
        except ImportError:
            pytest.skip("LPIPS not available")

        x = torch.rand(1, 3, 64, 64)
        S = torch.ones(1, 1, 64, 64) * 0.5  # Uniform saliency
        radius = 5.0

        # Test different alpha values
        alphas = [0.5, 1.0, 1.5, 2.0]
        lpips_model = lpips.LPIPS(net="alex")
        lpips_scores = []

        for alpha in alphas:
            x_adv = apply_field(x, S, alpha, radius)
            # LPIPS expects [-1, 1] range
            x_norm = 2 * x - 1
            x_adv_norm = 2 * x_adv - 1
            score = lpips_model(x_norm, x_adv_norm).item()
            lpips_scores.append(score)

        # Check monotonicity (allowing for small numerical variations)
        for i in range(1, len(lpips_scores)):
            assert (
                lpips_scores[i] >= lpips_scores[i - 1] - 1e-4
            ), f"LPIPS not monotonic at alpha {alphas[i]}"

    def test_field_input_validation(self):
        """Test input validation."""
        x = torch.rand(2, 3, 16, 16)

        # Wrong saliency dimensions
        S_wrong = torch.rand(2, 2, 16, 16)  # Should be (2, 1, 16, 16)
        with pytest.raises(ValueError, match="Expected 4D tensors"):
            apply_field(torch.rand(3, 16, 16), S_wrong, 1.0, 5.0)

        # Mismatched batch sizes
        S_wrong_batch = torch.rand(1, 1, 16, 16)  # Wrong batch size
        with pytest.raises(ValueError, match="Batch/spatial dimensions mismatch"):
            apply_field(x, S_wrong_batch, 1.0, 5.0)

        # Mismatched spatial dimensions
        S_wrong_spatial = torch.rand(2, 1, 8, 8)  # Wrong spatial size
        with pytest.raises(ValueError, match="Batch/spatial dimensions mismatch"):
            apply_field(x, S_wrong_spatial, 1.0, 5.0)

    def test_field_with_smoothing(self):
        """Test field application with saliency smoothing."""
        x = torch.rand(1, 3, 32, 32)
        S = torch.rand(1, 1, 32, 32)

        x_adv = field_with_smoothing(x, S, alpha=1.0, radius=5.0, smooth_radius=2.0)

        assert x_adv.shape == x.shape
        assert torch.all(x_adv >= 0.0) and torch.all(x_adv <= 1.0)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        validate_field_params(1.0, 5.0)

        # Invalid alpha
        with pytest.raises(ValueError, match="Alpha must be non-negative"):
            validate_field_params(-1.0, 5.0)

        # Invalid radius
        with pytest.raises(ValueError, match="Radius must be positive"):
            validate_field_params(1.0, 0.0)

        # Non-numeric parameters
        with pytest.raises(ValueError, match="Alpha must be numeric"):
            validate_field_params("invalid", 5.0)

        with pytest.raises(ValueError, match="Radius must be numeric"):
            validate_field_params(1.0, "invalid")

    def test_safe_wrapper(self):
        """Test safe wrapper function."""
        x = torch.rand(1, 3, 16, 16)
        S = torch.rand(1, 1, 16, 16)

        # Valid parameters
        x_adv = apply_field_safe(x, S, 1.0, 5.0)
        assert x_adv.shape == x.shape

        # Invalid parameters should raise
        with pytest.raises(ValueError):
            apply_field_safe(x, S, -1.0, 5.0)


class TestTVLoss:
    """Tests for Total Variation loss."""

    def test_tv_loss_computation(self):
        """Test TV loss computation."""
        # Create smooth field (low TV)
        smooth_field = torch.ones(2, 1, 16, 16) * 0.5
        smooth_tv = compute_tv_loss(smooth_field)

        # Create noisy field (high TV)
        noisy_field = torch.rand(2, 1, 16, 16)
        noisy_tv = compute_tv_loss(noisy_field)

        # Smooth field should have lower TV
        assert smooth_tv < noisy_tv
        assert smooth_tv >= 0.0
        assert noisy_tv >= 0.0

    def test_tv_loss_reductions(self):
        """Test different TV loss reductions."""
        field = torch.rand(3, 1, 8, 8)

        tv_mean = compute_tv_loss(field, reduction="mean")
        tv_sum = compute_tv_loss(field, reduction="sum")
        tv_none = compute_tv_loss(field, reduction="none")

        # Check relationships
        assert torch.allclose(tv_sum, tv_mean * 3, atol=1e-6)  # sum = mean * batch_size
        assert tv_none.shape[0] == 3  # Should return per-batch

        # Invalid reduction
        with pytest.raises(ValueError, match="Invalid reduction"):
            compute_tv_loss(field, reduction="invalid")

    def test_tv_loss_validation(self):
        """Test TV loss input validation."""
        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            compute_tv_loss(torch.rand(16, 16))


class TestFieldStats:
    """Tests for field statistics computation."""

    def test_field_stats_computation(self):
        """Test field statistics computation."""
        x_orig = torch.rand(2, 3, 16, 16)
        x_adv = x_orig + 0.1 * torch.rand_like(x_orig)  # Add perturbation

        stats = field_stats(x_orig, x_adv)

        # Check that all expected keys are present
        expected_keys = [
            "l2_norm",
            "linf_norm",
            "mean_abs_diff",
            "max_abs_diff",
            "affected_pixels",
        ]
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], float)
            assert stats[key] >= 0.0

        # L∞ norm should be >= mean absolute difference
        assert stats["linf_norm"] >= stats["mean_abs_diff"]

    def test_field_stats_identical_images(self):
        """Test stats for identical images."""
        x = torch.rand(1, 3, 8, 8)
        stats = field_stats(x, x)

        # All differences should be zero
        assert stats["l2_norm"] == 0.0
        assert stats["linf_norm"] == 0.0
        assert stats["mean_abs_diff"] == 0.0
        assert stats["max_abs_diff"] == 0.0
        assert stats["affected_pixels"] == 0.0


class TestFieldBoundaryConditions:
    """Tests for field boundary conditions and edge cases."""

    def test_extreme_alpha_values(self):
        """Test field with extreme alpha values."""
        x = torch.rand(1, 3, 16, 16)
        S = torch.rand(1, 1, 16, 16)
        radius = 3.0

        # Very small alpha vs large alpha comparison
        x_adv_small = apply_field(x, S, alpha=1e-6, radius=radius)
        diff_small = torch.norm(x_adv_small - x)

        # Large alpha (should create even larger difference)
        x_adv_large = apply_field(x, S, alpha=10.0, radius=radius)
        diff_large = torch.norm(x_adv_large - x)

        # Small alpha should create smaller difference than large alpha
        assert (
            diff_small < diff_large
        ), f"Small alpha diff: {diff_small}, Large alpha diff: {diff_large}"
        assert diff_large > 1.0  # Large alpha should create significant change

    def test_extreme_radius_values(self):
        """Test field with extreme radius values."""
        x = torch.rand(1, 3, 16, 16)
        S = torch.ones(1, 1, 16, 16)  # Uniform saliency
        alpha = 1.0

        # Very small radius (minimal blur)
        x_adv_small = apply_field(x, S, alpha, radius=0.1)

        # Large radius (heavy blur)
        x_adv_large = apply_field(x, S, alpha, radius=20.0)

        # Both should be valid
        assert torch.all(torch.isfinite(x_adv_small))
        assert torch.all(torch.isfinite(x_adv_large))
        assert torch.all(x_adv_small >= 0.0) and torch.all(x_adv_small <= 1.0)
        assert torch.all(x_adv_large >= 0.0) and torch.all(x_adv_large <= 1.0)

    def test_extreme_saliency_values(self):
        """Test field with extreme saliency values."""
        x = torch.rand(1, 3, 16, 16)
        alpha = 1.0
        radius = 5.0

        # Zero saliency (should be close to identity)
        S_zero = torch.zeros(1, 1, 16, 16)
        x_adv_zero = apply_field(x, S_zero, alpha, radius)
        assert torch.allclose(x_adv_zero, x, atol=1e-3)

        # Maximum saliency
        S_max = torch.ones(1, 1, 16, 16)
        x_adv_max = apply_field(x, S_max, alpha, radius)
        assert torch.all(torch.isfinite(x_adv_max))

    def test_small_images(self):
        """Test field on very small images."""
        x = torch.rand(1, 3, 4, 4)  # Very small image
        S = torch.rand(1, 1, 4, 4)

        x_adv = apply_field(x, S, alpha=1.0, radius=2.0)

        assert x_adv.shape == x.shape
        assert torch.all(torch.isfinite(x_adv))

    def test_single_pixel_saliency(self):
        """Test field with saliency concentrated in single pixel."""
        x = torch.rand(1, 3, 16, 16)
        S = torch.zeros(1, 1, 16, 16)
        S[0, 0, 8, 8] = 1.0  # Single pixel saliency

        x_adv = apply_field(x, S, alpha=2.0, radius=3.0)

        assert torch.all(torch.isfinite(x_adv))
        assert torch.all(x_adv >= 0.0) and torch.all(x_adv <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
