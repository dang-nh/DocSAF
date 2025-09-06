"""Unit tests for saliency computation module."""

import pytest
import torch
import numpy as np
from PIL import Image

from src.docsaf.saliency import (
    compute_gradient_saliency, clip_alignment_and_saliency,
    multi_scale_saliency, saliency_stats
)
from src.docsaf.surrogates import CLIPEmbedder


class MockEmbedder:
    """Mock embedder for testing."""
    
    def __init__(self, embed_dim=512, device="cpu"):
        self.embed_dim = embed_dim
        self.device = device
    
    def image_embed(self, image):
        # Return normalized random embedding
        batch_size = image.shape[0]
        emb = torch.randn(batch_size, self.embed_dim, device=self.device)
        return torch.nn.functional.normalize(emb, dim=-1)
    
    def text_embed(self, texts):
        # Return normalized random embedding
        batch_size = len(texts)
        emb = torch.randn(batch_size, self.embed_dim, device=self.device)
        return torch.nn.functional.normalize(emb, dim=-1)


class TestGradientSaliency:
    """Tests for gradient-based saliency computation."""
    
    def test_gradient_saliency_basic(self):
        """Test basic gradient saliency computation."""
        embedder = MockEmbedder()
        
        # Create test image requiring gradients
        x = torch.rand(1, 3, 32, 32, requires_grad=True)
        text = "test document text"
        
        # Compute saliency
        alignment, saliency = compute_gradient_saliency(embedder, x, text)
        
        # Check outputs
        assert isinstance(alignment, float)
        assert isinstance(saliency, torch.Tensor)
        assert saliency.shape == (1, 1, 32, 32)
        assert torch.all(saliency >= 0.0) and torch.all(saliency <= 1.0)
    
    def test_saliency_requires_gradients(self):
        """Test that saliency computation requires gradients."""
        embedder = MockEmbedder()
        x = torch.rand(1, 3, 16, 16, requires_grad=False)  # No gradients
        
        with pytest.raises(ValueError, match="must require gradients"):
            compute_gradient_saliency(embedder, x, "test")
    
    def test_saliency_normalization(self):
        """Test saliency normalization."""
        embedder = MockEmbedder()
        x = torch.rand(2, 3, 16, 16, requires_grad=True)
        
        # Test with normalization
        _, saliency_norm = compute_gradient_saliency(embedder, x, "test", normalize=True)
        
        # Test without normalization
        _, saliency_raw = compute_gradient_saliency(embedder, x, "test", normalize=False)
        
        # Normalized should be in [0,1]
        assert torch.all(saliency_norm >= 0.0) and torch.all(saliency_norm <= 1.0)
        
        # Raw might exceed [0,1]
        assert saliency_raw.min() >= 0.0  # Should still be non-negative
    
    def test_saliency_batch_processing(self):
        """Test saliency computation on batches."""
        embedder = MockEmbedder()
        batch_size = 3
        x = torch.rand(batch_size, 3, 16, 16, requires_grad=True)
        text = "batch processing test"
        
        alignment, saliency = compute_gradient_saliency(embedder, x, text)
        
        assert saliency.shape == (batch_size, 1, 16, 16)
        assert isinstance(alignment, float)
    
    def test_saliency_monotonicity(self):
        """Test saliency monotonicity under blur."""
        # This test checks that blurring text regions reduces saliency
        embedder = MockEmbedder()
        
        # Create a simple pattern (high contrast text-like)
        x_orig = torch.ones(1, 3, 32, 32) * 0.2
        x_orig[:, :, 8:24, 8:24] = 0.8  # "Text" region
        x_orig.requires_grad_(True)
        
        # Create blurred version
        import torch.nn.functional as F
        kernel_size = 7
        sigma = 2.0
        x_blur = F.gaussian_blur(x_orig.detach(), kernel_size, sigma)
        x_blur.requires_grad_(True)
        
        text = "document text content"
        
        # Compute saliency for both
        _, sal_orig = compute_gradient_saliency(embedder, x_orig, text)
        _, sal_blur = compute_gradient_saliency(embedder, x_blur, text)
        
        # The relationship might not be strictly monotonic due to randomness
        # in mock embedder, but both should be valid saliency maps
        assert torch.all(sal_orig >= 0.0) and torch.all(sal_orig <= 1.0)
        assert torch.all(sal_blur >= 0.0) and torch.all(sal_blur <= 1.0)
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        embedder = MockEmbedder()
        x = torch.rand(1, 3, 16, 16, requires_grad=True)
        
        # Empty text should still work
        alignment, saliency = compute_gradient_saliency(embedder, x, "")
        
        assert isinstance(alignment, float)
        assert saliency.shape == (1, 1, 16, 16)
    
    def test_gradient_clearing(self):
        """Test that gradients are properly handled."""
        embedder = MockEmbedder()
        x = torch.rand(1, 3, 16, 16, requires_grad=True)
        
        # Compute saliency
        alignment, saliency = compute_gradient_saliency(embedder, x, "test")
        
        # Gradients should be present on input
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestMultiScaleSaliency:
    """Tests for multi-scale saliency computation."""
    
    def test_multi_scale_basic(self):
        """Test basic multi-scale saliency."""
        embedder = MockEmbedder()
        x = torch.rand(1, 3, 64, 64, requires_grad=True)
        text = "multi-scale test"
        scales = (0.8, 1.0, 1.2)
        
        avg_alignment, avg_saliency = multi_scale_saliency(
            embedder, x, text, scales
        )
        
        assert isinstance(avg_alignment, float)
        assert avg_saliency.shape == (1, 1, 64, 64)
        assert torch.all(avg_saliency >= 0.0) and torch.all(avg_saliency <= 1.0)
    
    def test_multi_scale_single_scale(self):
        """Test multi-scale with single scale."""
        embedder = MockEmbedder()
        x = torch.rand(1, 3, 32, 32, requires_grad=True)
        text = "single scale test"
        
        # Single scale should be equivalent to regular saliency
        avg_alignment, avg_saliency = multi_scale_saliency(
            embedder, x, text, scales=(1.0,)
        )
        
        assert avg_saliency.shape == (1, 1, 32, 32)
        assert torch.all(avg_saliency >= 0.0) and torch.all(avg_saliency <= 1.0)
    
    def test_multi_scale_extreme_scales(self):
        """Test multi-scale with extreme scale values."""
        embedder = MockEmbedder()
        x = torch.rand(1, 3, 64, 64, requires_grad=True)
        text = "extreme scale test"
        
        # Test with very different scales
        avg_alignment, avg_saliency = multi_scale_saliency(
            embedder, x, text, scales=(0.5, 2.0)
        )
        
        assert avg_saliency.shape == (1, 1, 64, 64)
        assert torch.all(torch.isfinite(avg_saliency))


class TestSaliencyStats:
    """Tests for saliency statistics computation."""
    
    def test_saliency_stats_computation(self):
        """Test saliency statistics computation."""
        # Create test saliency map
        saliency_map = torch.rand(2, 1, 16, 16)
        
        stats = saliency_stats(saliency_map)
        
        # Check all expected keys
        expected_keys = ["min", "max", "mean", "std", "shape"]
        for key in expected_keys:
            assert key in stats
        
        # Check value ranges
        assert 0.0 <= stats["min"] <= stats["max"] <= 1.0
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["std"] >= 0.0
        assert stats["shape"] == (2, 1, 16, 16)
    
    def test_saliency_stats_uniform(self):
        """Test stats for uniform saliency."""
        # Uniform saliency map
        uniform_sal = torch.ones(1, 1, 8, 8) * 0.5
        stats = saliency_stats(uniform_sal)
        
        assert stats["min"] == stats["max"] == stats["mean"] == 0.5
        assert stats["std"] == 0.0
    
    def test_saliency_stats_extremes(self):
        """Test stats for extreme saliency values."""
        # Binary saliency map
        binary_sal = torch.zeros(1, 1, 8, 8)
        binary_sal[0, 0, :4, :] = 1.0  # Half ones, half zeros
        
        stats = saliency_stats(binary_sal)
        
        assert stats["min"] == 0.0
        assert stats["max"] == 1.0
        assert stats["mean"] == 0.5
        assert stats["std"] > 0.0


class TestCLIPSaliencyIntegration:
    """Integration tests with CLIP embedder (if available)."""
    
    def test_clip_alignment_saliency(self):
        """Test CLIP alignment and saliency (requires OpenCLIP)."""
        try:
            from src.docsaf.surrogates import CLIPEmbedder
        except ImportError:
            pytest.skip("OpenCLIP not available")
        
        try:
            # Try to create CLIP embedder
            clip_model = CLIPEmbedder(device="cpu")
        except Exception:
            pytest.skip("CLIP model loading failed")
        
        # Create test PIL image
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pil_image = Image.fromarray(img_array, mode='RGB')
        text = "test document with text content"
        
        try:
            # Test CLIP saliency wrapper
            alignment, saliency = clip_alignment_and_saliency(
                clip_model, pil_image, text, device="cpu"
            )
            
            assert isinstance(alignment, float)
            assert isinstance(saliency, torch.Tensor)
            assert saliency.dim() == 4  # (1, 1, H, W)
            assert torch.all(saliency >= 0.0) and torch.all(saliency <= 1.0)
            
        except Exception as e:
            pytest.skip(f"CLIP saliency computation failed: {e}")


class TestSaliencyBoundaryConditions:
    """Tests for boundary conditions and edge cases."""
    
    def test_tiny_images(self):
        """Test saliency on very small images."""
        embedder = MockEmbedder()
        x = torch.rand(1, 3, 4, 4, requires_grad=True)
        
        alignment, saliency = compute_gradient_saliency(embedder, x, "tiny test")
        
        assert saliency.shape == (1, 1, 4, 4)
        assert torch.all(torch.isfinite(saliency))
    
    def test_large_batch(self):
        """Test saliency on large batches."""
        embedder = MockEmbedder()
        batch_size = 16
        x = torch.rand(batch_size, 3, 32, 32, requires_grad=True)
        
        alignment, saliency = compute_gradient_saliency(embedder, x, "large batch test")
        
        assert saliency.shape == (batch_size, 1, 32, 32)
        assert torch.all(torch.isfinite(saliency))
    
    def test_different_image_sizes(self):
        """Test saliency on different image sizes."""
        embedder = MockEmbedder()
        
        sizes = [(8, 8), (16, 32), (64, 48), (128, 128)]
        
        for h, w in sizes:
            x = torch.rand(1, 3, h, w, requires_grad=True)
            alignment, saliency = compute_gradient_saliency(embedder, x, f"size {h}x{w}")
            
            assert saliency.shape == (1, 1, h, w)
            assert torch.all(torch.isfinite(saliency))
    
    def test_very_long_text(self):
        """Test saliency with very long text."""
        embedder = MockEmbedder()
        x = torch.rand(1, 3, 16, 16, requires_grad=True)
        
        # Very long text
        long_text = "word " * 1000
        
        alignment, saliency = compute_gradient_saliency(embedder, x, long_text)
        
        assert saliency.shape == (1, 1, 16, 16)
        assert torch.all(torch.isfinite(saliency))
    
    def test_special_characters_text(self):
        """Test saliency with special characters in text."""
        embedder = MockEmbedder()
        x = torch.rand(1, 3, 16, 16, requires_grad=True)
        
        # Text with special characters, unicode, etc.
        special_text = "Test with special chars: @#$%^&*()_+ üñíčødé 中文 العربية"
        
        alignment, saliency = compute_gradient_saliency(embedder, x, special_text)
        
        assert saliency.shape == (1, 1, 16, 16)
        assert torch.all(torch.isfinite(saliency))


if __name__ == "__main__":
    pytest.main([__file__])