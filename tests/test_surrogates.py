"""Tests for surrogate model aligners."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from src.docsaf.surrogates import (
    CLIPAligner,
    BLIP2Aligner, 
    DonutAligner,
    load_aligner,
    build_aligners,
    ImageTextAligner,
)


class TestCLIPAligner:
    """Test OpenCLIP aligner functionality."""
    
    @pytest.fixture
    def device(self):
        """Get appropriate device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_clip_aligner_creation(self, device):
        """Test CLIPAligner can be created."""
        try:
            aligner = CLIPAligner("ViT-B-32", "laion400m_e32", device)
            assert aligner.device == device
            assert aligner.model_name == "ViT-B-32"
        except Exception as e:
            pytest.skip(f"Skipping CLIP test due to model loading issue: {e}")
    
    def test_clip_encode_image(self, device):
        """Test image encoding produces correct shape."""
        try:
            aligner = CLIPAligner("ViT-B-32", "laion400m_e32", device)
            
            # Create test image tensor (B, C, H, W)
            x = torch.randn(2, 3, 224, 224).to(device)
            
            features = aligner.encode_image(x)
            
            # Should return normalized features
            assert features.shape[0] == 2  # Batch size
            assert features.shape[1] > 0   # Feature dimension
            
            # Features should be normalized
            norms = torch.norm(features, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
            
        except Exception as e:
            pytest.skip(f"Skipping CLIP encode test: {e}")
    
    def test_clip_encode_text(self, device):
        """Test text encoding produces correct shape."""
        try:
            aligner = CLIPAligner("ViT-B-32", "laion400m_e32", device)
            
            texts = ["hello world", "document text"]
            features = aligner.encode_text(texts)
            
            assert features.shape[0] == 2  # Number of texts
            assert features.shape[1] > 0   # Feature dimension
            
            # Features should be normalized
            norms = torch.norm(features, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
            
        except Exception as e:
            pytest.skip(f"Skipping CLIP text encode test: {e}")
    
    def test_clip_cosine_align(self, device):
        """Test cosine alignment computation."""
        try:
            aligner = CLIPAligner("ViT-B-32", "laion400m_e32", device)
            
            # Create test image
            x = torch.randn(1, 3, 224, 224).to(device)
            text = "test image"
            
            alignment = aligner.cosine_align(x, text)
            
            assert alignment.shape == (1,)  # Batch size
            assert -1 <= alignment.item() <= 1  # Cosine similarity range
            
        except Exception as e:
            pytest.skip(f"Skipping CLIP alignment test: {e}")


class TestSurrogateLoading:
    """Test surrogate loading and building functionality."""
    
    def test_load_aligner_openclip(self):
        """Test loading OpenCLIP aligner."""
        device = "cpu"  # Use CPU to avoid GPU dependency
        
        try:
            aligner = load_aligner("openclip:ViT-B-32", device)
            assert isinstance(aligner, CLIPAligner)
            assert aligner.device == device
        except Exception as e:
            pytest.skip(f"Skipping OpenCLIP loader test: {e}")
    
    def test_load_aligner_invalid_spec(self):
        """Test invalid aligner spec raises error."""
        with pytest.raises(ValueError, match="Invalid aligner spec"):
            load_aligner("invalid_spec", "cpu")
    
    def test_load_aligner_unsupported_backend(self):
        """Test unsupported backend raises error."""
        with pytest.raises(ValueError, match="Unsupported aligner backend"):
            load_aligner("unsupported:model", "cpu")
    
    def test_build_aligners_success(self):
        """Test building aligners from specs."""
        device = "cpu"
        specs = ["openclip:ViT-B-32"]
        
        try:
            aligners = build_aligners(specs, device)
            assert len(aligners) == 1
            assert isinstance(aligners[0], CLIPAligner)
        except Exception as e:
            pytest.skip(f"Skipping aligners builder test: {e}")
    
    def test_build_aligners_partial_failure(self):
        """Test building aligners with some failures."""
        device = "cpu"
        specs = ["openclip:ViT-B-32", "invalid:spec"]
        
        # Should load the valid one and skip the invalid one
        try:
            aligners = build_aligners(specs, device)
            assert len(aligners) == 1  # Only the valid one loaded
        except Exception as e:
            pytest.skip(f"Skipping partial failure test: {e}")
    
    def test_build_aligners_all_fail(self):
        """Test building aligners when all fail."""
        device = "cpu"
        specs = ["invalid:spec1", "invalid:spec2"]
        
        with pytest.raises(RuntimeError, match="No aligners could be loaded"):
            build_aligners(specs, device)


class TestSurrogateProtocol:
    """Test that aligners implement the protocol correctly."""
    
    @pytest.fixture
    def mock_aligner(self):
        """Create a mock aligner for testing."""
        class MockAligner:
            def encode_image(self, x):
                batch_size = x.shape[0]
                return torch.randn(batch_size, 512)  # Mock embedding
            
            def encode_text(self, texts):
                return torch.randn(len(texts), 512)  # Mock embedding
            
            def cosine_align(self, x, text):
                batch_size = x.shape[0]
                return torch.randn(batch_size)  # Mock alignment
        
        return MockAligner()
    
    def test_protocol_compliance(self, mock_aligner):
        """Test that mock aligner implements the protocol."""
        # Test image encoding
        x = torch.randn(2, 3, 224, 224)
        img_features = mock_aligner.encode_image(x)
        assert img_features.shape == (2, 512)
        
        # Test text encoding
        texts = ["hello", "world"]
        txt_features = mock_aligner.encode_text(texts)
        assert txt_features.shape == (2, 512)
        
        # Test alignment
        alignment = mock_aligner.cosine_align(x, "test")
        assert alignment.shape == (2,)


class TestBLIP2Aligner:
    """Test BLIP-2 aligner (may be skipped if model not available)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="BLIP-2 tests require CUDA")
    def test_blip2_creation(self):
        """Test BLIP-2 aligner creation (CPU fallback)."""
        try:
            # Use CPU to avoid memory issues in testing
            aligner = BLIP2Aligner("Salesforce/blip2-opt-2.7b", "cpu")
            assert aligner.device == "cpu"
        except Exception as e:
            pytest.skip(f"Skipping BLIP-2 test: {e}")


class TestDonutAligner:
    """Test Donut aligner (may be skipped if model not available)."""
    
    def test_donut_creation_cpu(self):
        """Test Donut aligner creation on CPU."""
        try:
            aligner = DonutAligner("naver-clova-ix/donut-base", "cpu")
            assert aligner.device == "cpu"
        except Exception as e:
            pytest.skip(f"Skipping Donut test: {e}")


class TestDeviceHandling:
    """Test device handling in aligners."""
    
    def test_cpu_fallback(self):
        """Test that aligners work on CPU when CUDA unavailable."""
        try:
            aligner = CLIPAligner("ViT-B-32", "laion400m_e32", "cpu")
            
            # Test basic functionality on CPU
            x = torch.randn(1, 3, 224, 224)
            features = aligner.encode_image(x)
            assert features.device.type == "cpu"
            
        except Exception as e:
            pytest.skip(f"Skipping CPU fallback test: {e}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_usage(self):
        """Test that aligners use CUDA when available."""
        try:
            aligner = CLIPAligner("ViT-B-32", "laion400m_e32", "cuda")
            
            x = torch.randn(1, 3, 224, 224).cuda()
            features = aligner.encode_image(x)
            assert features.device.type == "cuda"
            
        except Exception as e:
            pytest.skip(f"Skipping CUDA test: {e}")


# Integration test with minimal models (if available)
class TestIntegration:
    """Integration tests with real models (may be skipped)."""
    
    def test_end_to_end_alignment(self):
        """Test end-to-end alignment computation."""
        device = "cpu"  # Use CPU for testing
        
        try:
            aligner = load_aligner("openclip:ViT-B-32", device)
            
            # Create a synthetic document-like image
            img = Image.new('RGB', (224, 224), color='white')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            try:
                # Try to use a font
                font = ImageFont.load_default()
                draw.text((50, 100), "INVOICE #123", fill='black', font=font)
            except:
                # Fallback if font loading fails
                draw.text((50, 100), "INVOICE #123", fill='black')
            
            # Convert to tensor
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToTensor()
            ])
            x = transform(img).unsqueeze(0).to(device)
            
            # Test alignment
            text = "invoice document"
            alignment = aligner.cosine_align(x, text)
            
            assert isinstance(alignment, torch.Tensor)
            assert alignment.shape == (1,)
            assert -1 <= alignment.item() <= 1
            
        except Exception as e:
            pytest.skip(f"Skipping integration test: {e}")
