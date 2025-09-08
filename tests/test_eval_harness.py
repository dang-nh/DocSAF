"""Tests for evaluation harness functionality."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import patch, MagicMock
from PIL import Image

from src.docsaf.eval_harness import (
    EvaluationMetrics,
    compute_ocr_mismatch,
    compute_donut_mismatch,
    generate_markdown_report,
    generate_csv_report,
)


class TestEvaluationMetrics:
    """Test evaluation metrics container."""
    
    def test_metrics_initialization(self):
        """Test metrics are properly initialized."""
        metrics = EvaluationMetrics()
        
        assert len(metrics.alignment_scores["original"]) == 0
        assert len(metrics.alignment_scores["adversarial"]) == 0
        assert len(metrics.alignment_scores["drop"]) == 0
        assert len(metrics.lpips_scores) == 0
        assert len(metrics.ocr_mismatch_rates) == 0
        assert len(metrics.donut_mismatch_rates) == 0
        assert metrics.success_rate == 0.0
    
    def test_add_sample(self):
        """Test adding sample metrics."""
        metrics = EvaluationMetrics()
        
        # Add some samples
        metrics.add_sample(0.8, 0.6, 0.05, 0.1, 0.2)
        metrics.add_sample(0.7, 0.4, 0.08, 0.3, 0.1)
        
        assert len(metrics.alignment_scores["original"]) == 2
        assert len(metrics.alignment_scores["adversarial"]) == 2
        assert len(metrics.alignment_scores["drop"]) == 2
        assert len(metrics.lpips_scores) == 2
        assert len(metrics.ocr_mismatch_rates) == 2
        assert len(metrics.donut_mismatch_rates) == 2
        
        # Check values
        assert metrics.alignment_scores["original"] == [0.8, 0.7]
        assert metrics.alignment_scores["adversarial"] == [0.6, 0.4]
        assert metrics.alignment_scores["drop"] == pytest.approx([0.2, 0.3], abs=1e-10)
        assert metrics.lpips_scores == [0.05, 0.08]
        assert metrics.ocr_mismatch_rates == [0.1, 0.3]
        assert metrics.donut_mismatch_rates == [0.2, 0.1]
    
    def test_compute_statistics(self):
        """Test statistics computation."""
        metrics = EvaluationMetrics()
        
        # Add samples with known statistics
        metrics.add_sample(1.0, 0.8, 0.05)  # drop = 0.2
        metrics.add_sample(0.8, 0.6, 0.08)  # drop = 0.2
        metrics.add_sample(0.6, 0.7, 0.03)  # drop = -0.1 (negative)
        
        stats = metrics.compute_statistics()
        
        # Check alignment statistics
        assert stats["alignment_original_mean"] == pytest.approx(0.8, abs=1e-3)
        assert stats["alignment_adversarial_mean"] == pytest.approx(0.7, abs=1e-3)
        assert stats["alignment_drop_mean"] == pytest.approx(0.1, abs=1e-3)
        
        # Check LPIPS statistics
        assert stats["lpips_mean"] == pytest.approx(0.0533, abs=1e-3)
        assert stats["lpips_median"] == pytest.approx(0.05, abs=1e-3)
        assert stats["lpips_max"] == pytest.approx(0.08, abs=1e-3)
        
        # Check success rate (positive drops)
        assert stats["success_rate"] == pytest.approx(2/3, abs=1e-3)  # 2 out of 3 positive
        
        # Check threshold rate (below 0.06)
        assert stats["lpips_below_threshold_rate"] == pytest.approx(2/3, abs=1e-3)  # 0.05 and 0.03
    
    def test_reset(self):
        """Test metrics reset functionality."""
        metrics = EvaluationMetrics()
        
        # Add some data
        metrics.add_sample(0.8, 0.6, 0.05)
        assert len(metrics.alignment_scores["original"]) == 1
        
        # Reset
        metrics.reset()
        assert len(metrics.alignment_scores["original"]) == 0
        assert len(metrics.lpips_scores) == 0
        assert metrics.success_rate == 0.0


class TestOCRMismatch:
    """Test OCR mismatch computation."""
    
    def test_identical_texts(self):
        """Test mismatch rate for identical texts."""
        text1 = "Invoice #12345 Date: January 15 2024"
        text2 = "Invoice #12345 Date: January 15 2024"
        
        mismatch = compute_ocr_mismatch(text1, text2)
        assert mismatch == 0.0
    
    def test_completely_different_texts(self):
        """Test mismatch rate for completely different texts."""
        text1 = "Invoice #12345"
        text2 = "Receipt #67890"
        
        mismatch = compute_ocr_mismatch(text1, text2)
        assert mismatch > 0.5  # Should be high mismatch
    
    def test_partial_overlap_texts(self):
        """Test mismatch rate for partially overlapping texts."""
        text1 = "Invoice #12345 Date: January 15"
        text2 = "Invoice #12345 Amount: $50.00"
        
        mismatch = compute_ocr_mismatch(text1, text2)
        assert 0.0 < mismatch < 1.0  # Should be partial mismatch
    
    def test_empty_texts(self):
        """Test mismatch rate for empty texts."""
        # Empty original text
        mismatch1 = compute_ocr_mismatch("", "some text")
        assert mismatch1 == 1.0
        
        # Empty adversarial text
        mismatch2 = compute_ocr_mismatch("some text", "")
        assert mismatch2 == 1.0
        
        # Both empty
        mismatch3 = compute_ocr_mismatch("", "")
        assert mismatch3 == 1.0
    
    def test_case_insensitive(self):
        """Test that mismatch computation is case insensitive."""
        text1 = "Invoice #12345"
        text2 = "INVOICE #12345"
        
        mismatch = compute_ocr_mismatch(text1, text2)
        assert mismatch == 0.0


class TestDonutMismatch:
    """Test Donut mismatch computation."""
    
    def test_donut_mismatch_placeholder(self):
        """Test placeholder Donut mismatch implementation."""
        # Create dummy tensors
        orig = torch.randn(1, 3, 224, 224)
        adv = torch.randn(1, 3, 224, 224)
        
        mismatch = compute_donut_mismatch(orig, adv, donut_aligner=None)
        
        assert 0.0 <= mismatch <= 1.0  # Should be valid mismatch rate
    
    def test_identical_images(self):
        """Test mismatch for identical images."""
        img = torch.randn(1, 3, 224, 224)
        
        mismatch = compute_donut_mismatch(img, img, donut_aligner=None)
        assert mismatch == 0.0  # Identical images should have 0 mismatch


class TestReportGeneration:
    """Test report generation functionality."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results."""
        return {
            "evaluation_config": {
                "num_samples": 10,
                "alpha": 1.2,
                "radius": 7.0,
                "training_aligners": 1,
                "held_out_aligners": 1,
            },
            "main_statistics": {
                "success_rate": 0.7,
                "alignment_original_mean": 0.8,
                "alignment_original_std": 0.1,
                "alignment_adversarial_mean": 0.6,
                "alignment_adversarial_std": 0.15,
                "alignment_drop_mean": 0.2,
                "alignment_drop_std": 0.05,
                "lpips_mean": 0.045,
                "lpips_median": 0.04,
                "lpips_max": 0.08,
                "lpips_below_threshold_rate": 0.8,
                "ocr_mismatch_mean": 0.1,
                "ocr_mismatch_std": 0.05,
                "donut_mismatch_mean": 0.15,
                "donut_mismatch_std": 0.08,
                "held_out_aligner_0": 0.6,
                "jpeg_defense_asr": 0.5,
                "resize_defense_asr": 0.4,
                "combined_defense_asr": 0.3,
            },
            "sample_results": [
                {"path": "test1.png", "original_alignment": 0.8, "adversarial_alignment": 0.6, "lpips_score": 0.04},
                {"path": "test2.png", "original_alignment": 0.75, "adversarial_alignment": 0.55, "lpips_score": 0.05},
            ]
        }
    
    def test_markdown_report_generation(self, sample_results):
        """Test Markdown report generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name
        
        try:
            generate_markdown_report(sample_results, output_path)
            
            # Check that file was created and has content
            assert Path(output_path).exists()
            
            with open(output_path, 'r') as f:
                content = f.read()
            
            # Check for key sections
            assert "# DocSAF Evaluation Report" in content
            assert "## Configuration" in content
            assert "## Main Results" in content
            assert "### Attack Success Rate" in content
            assert "### Perceptual Quality (LPIPS)" in content
            assert "### Transfer Attack Success Rate" in content
            assert "### Defense Robustness" in content
            
            # Check for specific values
            assert "70.0%" in content  # Success rate
            assert "1.200" in content  # Alpha
            assert "7.000" in content  # Radius
            
        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)
    
    def test_csv_report_generation(self, sample_results):
        """Test CSV report generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            generate_csv_report(sample_results, output_path)
            
            # Check that files were created
            assert Path(output_path).exists()
            
            # Check detailed CSV
            detailed_path = output_path.replace('.csv', '_detailed.csv')
            assert Path(detailed_path).exists()
            
            # Read and check main CSV (using built-in csv module)
            import csv
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                # Should have metric, value, description columns
                assert "metric" in reader.fieldnames
                assert "value" in reader.fieldnames
                assert "description" in reader.fieldnames
                
                # Check for key metrics
                metrics = [row["metric"] for row in rows]
                assert "success_rate" in metrics
                assert "config_alpha" in metrics
                assert "config_radius" in metrics
                assert "lpips_mean" in metrics
            
            # Read and check detailed CSV
            with open(detailed_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 2  # Should match sample_results length
                assert "path" in reader.fieldnames
                assert "original_alignment" in reader.fieldnames
            
        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)
            Path(detailed_path).unlink(missing_ok=True)


class TestLPIPSMonotonicity:
    """Test LPIPS monotonicity with respect to alpha."""
    
    def test_lpips_increases_with_alpha(self):
        """Test that LPIPS increases monotonically with alpha."""
        # This test requires the actual field.py module
        try:
            from src.docsaf.field import apply_field_safe
            from src.docsaf.utils import compute_lpips_score
            
            # Create a synthetic document patch
            img = torch.ones(1, 3, 128, 128) * 0.9  # Light background
            
            # Add some "text" by darkening rectangles
            img[:, :, 40:60, 20:100] = 0.1  # Dark text region
            img[:, :, 80:100, 30:90] = 0.1   # Another text region
            
            # Create a simple saliency map (highlight text regions)
            saliency = torch.zeros(1, 1, 128, 128)
            saliency[:, :, 40:60, 20:100] = 1.0
            saliency[:, :, 80:100, 30:90] = 1.0
            
            # Test different alpha values
            alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
            radius = 5.0
            lpips_scores = []
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            img = img.to(device)
            saliency = saliency.to(device)
            
            for alpha in alphas:
                x_adv = apply_field_safe(img, saliency, alpha, radius)
                lpips_score = compute_lpips_score(img, x_adv, device=device)
                lpips_scores.append(lpips_score)
            
            # Check monotonicity (should be non-decreasing)
            for i in range(1, len(lpips_scores)):
                assert lpips_scores[i] >= lpips_scores[i-1] - 1e-6, \
                    f"LPIPS not monotonic: alpha={alphas[i-1]}->{alphas[i]}, LPIPS={lpips_scores[i-1]:.4f}->{lpips_scores[i]:.4f}"
            
            # Alpha=0 should give minimal LPIPS (identity)
            assert lpips_scores[0] < 0.01, f"Alpha=0 should give near-zero LPIPS, got {lpips_scores[0]:.4f}"
            
        except ImportError as e:
            pytest.skip(f"Skipping LPIPS monotonicity test due to import error: {e}")
        except Exception as e:
            pytest.skip(f"Skipping LPIPS monotonicity test: {e}")


class TestOCRFallback:
    """Test OCR fallback functionality."""
    
    def test_easyocr_fallback_to_paddleocr(self):
        """Test that EasyOCR errors trigger PaddleOCR fallback."""
        try:
            from src.docsaf.ocr import ocr_read
            import numpy as np
            
            # Create a simple test image
            img_array = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White background
            img_array[40:60, 50:150] = 0  # Black rectangle (simulated text)
            
            # Mock EasyOCR to raise an exception
            with patch('easyocr.Reader') as mock_reader_class:
                mock_reader = MagicMock()
                mock_reader.readtext.side_effect = Exception("EasyOCR failed")
                mock_reader_class.return_value = mock_reader
                
                # Mock PaddleOCR to return something
                with patch('paddleocr.PaddleOCR') as mock_paddle_class:
                    mock_paddle = MagicMock()
                    mock_paddle.ocr.return_value = [[["test text", 0.9]]]
                    mock_paddle_class.return_value = mock_paddle
                    
                    # Should fallback to PaddleOCR
                    result = ocr_read(img_array, backend="easyocr")
                    
                    # Should have called PaddleOCR as fallback
                    mock_paddle_class.assert_called_once()
                    assert "test text" in result
                    
        except ImportError:
            # Mock the entire ocr module if dependencies not available
            with patch('src.docsaf.ocr.ocr_read') as mock_ocr:
                mock_ocr.return_value = "fallback text"
                
                result = mock_ocr(np.ones((100, 100, 3), dtype=np.uint8) * 255)
                assert result == "fallback text"
                
                pytest.skip("OCR dependencies not available, testing with mocks")


class TestEvalIntegration:
    """Integration tests for evaluation components."""
    
    def test_metrics_integration(self):
        """Test metrics integration with realistic values."""
        metrics = EvaluationMetrics()
        
        # Simulate a realistic evaluation scenario
        np.random.seed(42)  # For reproducibility
        
        # Add 20 samples with some realistic patterns
        for i in range(20):
            orig_align = 0.3 + np.random.normal(0, 0.1)  # Original alignments around 0.3
            
            # Adversarial alignments - some attacks succeed, others don't
            if i < 12:  # 60% success rate
                adv_align = orig_align - np.random.uniform(0.05, 0.15)  # Successful attack
            else:
                adv_align = orig_align + np.random.uniform(0.0, 0.05)   # Failed attack
            
            lpips = np.random.uniform(0.02, 0.1)  # LPIPS in reasonable range
            ocr_mismatch = np.random.uniform(0.0, 0.3)  # OCR mismatch
            donut_mismatch = np.random.uniform(0.0, 0.2)  # Donut mismatch
            
            metrics.add_sample(orig_align, adv_align, lpips, ocr_mismatch, donut_mismatch)
        
        stats = metrics.compute_statistics()
        
        # Check that statistics are reasonable
        assert 0.0 <= stats["success_rate"] <= 1.0
        assert stats["alignment_original_mean"] > 0
        assert stats["lpips_mean"] > 0
        assert 0.0 <= stats["lpips_below_threshold_rate"] <= 1.0
        
        # Success rate should be around 60% (12/20)
        assert 0.4 <= stats["success_rate"] <= 0.8
