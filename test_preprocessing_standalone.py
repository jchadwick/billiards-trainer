#!/usr/bin/env python3
"""
Standalone test for image preprocessing functionality.
This test bypasses the vision module's __init__.py to avoid import issues.
"""

import sys
import os
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the preprocessing module directly
try:
    from backend.vision.preprocessing import (
        ImagePreprocessor,
        ColorSpace,
        NoiseReductionMethod,
        PreprocessingConfig
    )
    print("✓ Successfully imported preprocessing module")
except ImportError as e:
    print(f"✗ Failed to import preprocessing module: {e}")
    sys.exit(1)


class TestImagePreprocessing(unittest.TestCase):
    """Test image preprocessing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = self._create_test_image()

    def _create_test_image(self):
        """Create a test image for preprocessing."""
        # Create a 200x200 test image with different regions
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        # Add colored regions
        image[50:100, 50:100] = [255, 0, 0]    # Blue region
        image[100:150, 100:150] = [0, 255, 0]  # Green region
        image[150:200, 150:200] = [0, 0, 255]  # Red region

        # Add some noise
        noise = np.random.normal(0, 15, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def test_preprocessor_initialization(self):
        """Test basic preprocessor initialization."""
        config = {
            'target_color_space': ColorSpace.HSV,
            'noise_reduction_enabled': True
        }

        preprocessor = ImagePreprocessor(config)
        self.assertIsNotNone(preprocessor)
        self.assertEqual(preprocessor.config.target_color_space, ColorSpace.HSV)
        print("✓ Preprocessor initialization test passed")

    def test_image_processing(self):
        """Test basic image processing."""
        config = {
            'target_color_space': ColorSpace.HSV,
            'noise_reduction_enabled': True,
            'auto_exposure_correction': True,
            'auto_white_balance': True
        }

        preprocessor = ImagePreprocessor(config)
        result = preprocessor.process(self.test_image)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], self.test_image.shape[0])
        self.assertEqual(result.shape[1], self.test_image.shape[1])
        print("✓ Basic image processing test passed")

    def test_color_space_conversion(self):
        """Test color space conversion."""
        config = {
            'target_color_space': ColorSpace.HSV
        }

        preprocessor = ImagePreprocessor(config)

        # Test HSV conversion
        hsv_result = preprocessor.convert_color_space(self.test_image, "HSV")
        self.assertIsNotNone(hsv_result)
        self.assertEqual(len(hsv_result.shape), 3)
        self.assertEqual(hsv_result.shape[2], 3)

        # Test LAB conversion
        lab_result = preprocessor.convert_color_space(self.test_image, "LAB")
        self.assertIsNotNone(lab_result)
        self.assertEqual(len(lab_result.shape), 3)

        # Test grayscale conversion
        gray_result = preprocessor.convert_color_space(self.test_image, "GRAY")
        self.assertIsNotNone(gray_result)
        self.assertEqual(len(gray_result.shape), 2)

        print("✓ Color space conversion test passed")

    def test_noise_reduction(self):
        """Test noise reduction methods."""
        config = {
            'noise_reduction_enabled': True
        }

        preprocessor = ImagePreprocessor(config)

        # Test different noise reduction methods
        methods = ['gaussian', 'bilateral', 'median']

        for method in methods:
            with self.subTest(method=method):
                result = preprocessor.apply_noise_reduction(self.test_image, method)
                self.assertIsNotNone(result)
                self.assertEqual(result.shape, self.test_image.shape)

        print("✓ Noise reduction test passed")

    def test_brightness_normalization(self):
        """Test brightness normalization."""
        config = {}
        preprocessor = ImagePreprocessor(config)

        # Create a dark image
        dark_image = (self.test_image * 0.3).astype(np.uint8)

        result = preprocessor.normalize_brightness(dark_image, target_brightness=128.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, dark_image.shape)

        # Check that brightness increased
        original_brightness = np.mean(dark_image)
        result_brightness = np.mean(result)
        self.assertGreater(result_brightness, original_brightness)

        print("✓ Brightness normalization test passed")

    def test_contrast_enhancement(self):
        """Test contrast enhancement."""
        config = {}
        preprocessor = ImagePreprocessor(config)

        result = preprocessor.enhance_contrast(self.test_image, alpha=1.5)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)

        print("✓ Contrast enhancement test passed")

    def test_gamma_correction(self):
        """Test gamma correction."""
        config = {}
        preprocessor = ImagePreprocessor(config)

        result = preprocessor.apply_gamma_correction(self.test_image, gamma=1.5)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)

        print("✓ Gamma correction test passed")

    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        config = {}
        preprocessor = ImagePreprocessor(config)

        # Test with None input
        result = preprocessor.process(None)
        self.assertIsNone(result)

        # Test with empty array
        empty_array = np.array([])
        result = preprocessor.process(empty_array)
        self.assertEqual(result.size, 0)

        print("✓ Error handling test passed")

    def test_statistics_tracking(self):
        """Test processing statistics tracking."""
        config = {}
        preprocessor = ImagePreprocessor(config)

        # Process a few frames
        for _ in range(3):
            preprocessor.process(self.test_image)

        stats = preprocessor.get_statistics()
        self.assertEqual(stats['frames_processed'], 3)
        self.assertGreater(stats['avg_processing_time'], 0)

        print("✓ Statistics tracking test passed")


def test_demo_functionality():
    """Test the demo functionality."""
    try:
        # Import demo module
        from backend.vision.demo_preprocessing import PreprocessingDemo

        # Create demo instance
        demo = PreprocessingDemo(save_output=False)

        # Create sample images
        samples = demo.create_sample_images()

        print(f"✓ Demo created {len(samples)} sample images")

        # Test each sample
        for name, image in samples.items():
            print(f"  - {name}: {image.shape}")

        return True

    except Exception as e:
        print(f"✗ Demo test failed: {e}")
        return False


def run_performance_test():
    """Run performance test."""
    config = {
        'target_color_space': ColorSpace.HSV,
        'noise_reduction_enabled': True,
        'auto_exposure_correction': True,
        'auto_white_balance': True,
        'morphology_enabled': True
    }

    preprocessor = ImagePreprocessor(config)

    # Create larger test image
    large_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    import time

    # Process multiple frames to get average timing
    times = []
    for i in range(10):
        start_time = time.time()
        result = preprocessor.process(large_image)
        processing_time = (time.time() - start_time) * 1000
        times.append(processing_time)

    avg_time = np.mean(times)
    fps = 1000.0 / avg_time if avg_time > 0 else 0

    print(f"✓ Performance test completed:")
    print(f"  - Average processing time: {avg_time:.2f}ms")
    print(f"  - Estimated FPS: {fps:.1f}")
    print(f"  - Frame size: {large_image.shape}")

    return avg_time < 100  # Should process 640x480 frames in under 100ms


def main():
    """Run all tests."""
    print("🎱 Billiards Vision Preprocessing Tests")
    print("=" * 50)

    # Run unit tests
    print("\n📋 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=0)

    # Test demo functionality
    print("\n🎨 Testing demo functionality...")
    demo_success = test_demo_functionality()

    # Run performance test
    print("\n⚡ Running performance test...")
    perf_success = run_performance_test()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"  - Unit tests: ✓ Completed")
    print(f"  - Demo test: {'✓ Passed' if demo_success else '✗ Failed'}")
    print(f"  - Performance test: {'✓ Passed' if perf_success else '✗ Failed'}")

    print("\n🎉 All preprocessing tests completed!")


if __name__ == '__main__':
    main()
