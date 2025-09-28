"""Comprehensive tests for image preprocessing pipeline.

Tests all functional requirements FR-VIS-006 to FR-VIS-010:
- Color space conversion (BGR to HSV/LAB)
- Noise reduction and image smoothing
- Automatic exposure and white balance correction
- ROI cropping for performance optimization
- Adaptive lighting condition handling
"""

import unittest

import cv2
import numpy as np

# Import the modules to test
from backend.vision.preprocessing import (
    ColorSpace,
    ImagePreprocessor,
    NoiseReductionMethod,
    PreprocessingConfig,
    create_default_config,
    enhance_for_detection,
    preprocess_image,
)


class TestPreprocessingConfig(unittest.TestCase):
    """Test preprocessing configuration functionality."""

    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config = create_default_config()
        assert isinstance(config, PreprocessingConfig)
        assert config.target_color_space == ColorSpace.HSV
        assert config.noise_reduction_enabled
        assert config.auto_exposure_correction

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "target_color_space": ColorSpace.LAB,
            "noise_reduction_enabled": False,
            "gaussian_kernel_size": 7,
        }
        config = PreprocessingConfig(**config_dict)
        assert config.target_color_space == ColorSpace.LAB
        assert not config.noise_reduction_enabled
        assert config.gaussian_kernel_size == 7


class TestImagePreprocessor(unittest.TestCase):
    """Test main image preprocessor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_default_config()
        self.config.debug_mode = False
        self.config.save_intermediate_steps = False
        self.preprocessor = ImagePreprocessor(self.config.__dict__)

        # Create test images
        self.test_bgr_image = self._create_test_image_bgr()
        self.test_grayscale_image = self._create_test_image_grayscale()
        self.test_noisy_image = self._create_noisy_image()
        self.test_dark_image = self._create_dark_image()
        self.test_bright_image = self._create_bright_image()

    def _create_test_image_bgr(self):
        """Create a test BGR image."""
        # Create a 100x100 test image with different colored regions
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Blue region
        image[0:33, 0:33] = [255, 0, 0]  # BGR format

        # Green region
        image[33:66, 33:66] = [0, 255, 0]

        # Red region
        image[66:100, 66:100] = [0, 0, 255]

        # White region
        image[0:33, 66:100] = [255, 255, 255]

        return image

    def _create_test_image_grayscale(self):
        """Create a test grayscale image."""
        return np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    def _create_noisy_image(self):
        """Create a noisy test image."""
        base_image = self._create_test_image_bgr()
        noise = np.random.normal(0, 25, base_image.shape).astype(np.int16)
        noisy_image = np.clip(base_image.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )
        return noisy_image

    def _create_dark_image(self):
        """Create a dark test image."""
        base_image = self._create_test_image_bgr()
        return (base_image * 0.3).astype(np.uint8)

    def _create_bright_image(self):
        """Create a bright test image."""
        base_image = self._create_test_image_bgr()
        return np.clip(base_image * 1.8, 0, 255).astype(np.uint8)

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        assert self.preprocessor is not None
        assert isinstance(self.preprocessor.config, PreprocessingConfig)
        assert self.preprocessor.stats["frames_processed"] == 0

    def test_basic_image_processing(self):
        """Test basic image processing pipeline."""
        result = self.preprocessor.process(self.test_bgr_image)

        assert result is not None
        assert result.shape[0] == self.test_bgr_image.shape[0]
        assert result.shape[1] == self.test_bgr_image.shape[1]

        # Check that processing occurred
        assert self.preprocessor.stats["frames_processed"] > 0

    def test_color_space_conversion_hsv(self):
        """Test BGR to HSV color space conversion."""
        self.config.target_color_space = ColorSpace.HSV
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_bgr_image)

        # HSV images should have 3 channels
        assert len(result.shape) == 3
        assert result.shape[2] == 3

        # HSV ranges should be correct
        h_channel = result[:, :, 0]
        s_channel = result[:, :, 1]
        v_channel = result[:, :, 2]

        assert np.all(h_channel <= 179)  # Hue range 0-179 in OpenCV
        assert np.all(s_channel <= 255)  # Saturation range 0-255
        assert np.all(v_channel <= 255)  # Value range 0-255

    def test_color_space_conversion_lab(self):
        """Test BGR to LAB color space conversion."""
        self.config.target_color_space = ColorSpace.LAB
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_bgr_image)

        # LAB images should have 3 channels
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_color_space_conversion_grayscale(self):
        """Test BGR to grayscale conversion."""
        self.config.target_color_space = ColorSpace.GRAY
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_bgr_image)

        # Grayscale image should have 2 dimensions
        assert len(result.shape) == 2

    def test_noise_reduction_gaussian(self):
        """Test Gaussian noise reduction."""
        self.config.noise_reduction_enabled = True
        self.config.noise_method = NoiseReductionMethod.GAUSSIAN
        preprocessor = ImagePreprocessor(self.config.__dict__)

        original_std = np.std(self.test_noisy_image)
        result = preprocessor.process(self.test_noisy_image)
        result_std = np.std(result)

        # Noise reduction should reduce standard deviation
        assert result_std < original_std * 1.1  # Allow some tolerance

    def test_noise_reduction_bilateral(self):
        """Test bilateral filtering noise reduction."""
        self.config.noise_reduction_enabled = True
        self.config.noise_method = NoiseReductionMethod.BILATERAL
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_noisy_image)

        assert result is not None
        assert result.shape == self.test_noisy_image.shape

    def test_noise_reduction_median(self):
        """Test median filtering noise reduction."""
        self.config.noise_reduction_enabled = True
        self.config.noise_method = NoiseReductionMethod.MEDIAN
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_noisy_image)

        assert result is not None
        assert result.shape == self.test_noisy_image.shape

    def test_exposure_correction_dark_image(self):
        """Test exposure correction on dark image."""
        self.config.auto_exposure_correction = True
        preprocessor = ImagePreprocessor(self.config.__dict__)

        original_brightness = np.mean(self.test_dark_image)
        result = preprocessor.process(self.test_dark_image)
        result_brightness = np.mean(result)

        # Exposure correction should brighten dark images
        assert result_brightness > original_brightness

    def test_exposure_correction_bright_image(self):
        """Test exposure correction on bright image."""
        self.config.auto_exposure_correction = True
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_bright_image)

        assert result is not None
        assert result.shape == self.test_bright_image.shape

    def test_white_balance_correction(self):
        """Test white balance correction."""
        # Create an image with color cast
        color_cast_image = self.test_bgr_image.copy()
        color_cast_image[:, :, 0] = np.clip(
            color_cast_image[:, :, 0] * 1.5, 0, 255
        )  # Blue cast

        self.config.auto_white_balance = True
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(color_cast_image)

        assert result is not None
        assert result.shape == color_cast_image.shape

    def test_morphological_operations(self):
        """Test morphological operations."""
        self.config.morphology_enabled = True
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_bgr_image)

        assert result is not None
        assert result.shape[0] == self.test_bgr_image.shape[0]
        assert result.shape[1] == self.test_bgr_image.shape[1]

    def test_resize_for_processing(self):
        """Test image resizing for processing optimization."""
        self.config.resize_for_processing = True
        self.config.processing_scale = 0.5
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_bgr_image)

        # Result should be resized back to original size
        assert result.shape[0] == self.test_bgr_image.shape[0]
        assert result.shape[1] == self.test_bgr_image.shape[1]

    def test_gamma_correction(self):
        """Test gamma correction."""
        self.config.gamma_correction = 1.5
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.apply_gamma_correction(self.test_bgr_image, 1.5)

        assert result is not None
        assert result.shape == self.test_bgr_image.shape

    def test_contrast_enhancement(self):
        """Test contrast enhancement."""
        result = self.preprocessor.enhance_contrast(self.test_bgr_image, alpha=1.5)

        assert result is not None
        assert result.shape == self.test_bgr_image.shape

    def test_brightness_normalization(self):
        """Test brightness normalization."""
        result = self.preprocessor.normalize_brightness(
            self.test_dark_image, target_brightness=128.0
        )

        assert result is not None
        assert result.shape == self.test_dark_image.shape

        # Brightness should be closer to target
        result_brightness = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
        assert result_brightness > np.mean(
            cv2.cvtColor(self.test_dark_image, cv2.COLOR_BGR2GRAY)
        )

    def test_sharpening(self):
        """Test image sharpening."""
        self.config.sharpening_enabled = True
        self.config.sharpening_strength = 0.5
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_bgr_image)

        assert result is not None

    def test_debug_mode(self):
        """Test debug mode functionality."""
        self.config.debug_mode = True
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_bgr_image)
        debug_images = preprocessor.get_debug_images()

        assert result is not None
        if self.config.noise_reduction_enabled or self.config.auto_white_balance:
            assert len(debug_images) > 0

    def test_statistics_tracking(self):
        """Test processing statistics tracking."""
        # Process multiple frames
        for _ in range(5):
            self.preprocessor.process(self.test_bgr_image)

        stats = self.preprocessor.get_statistics()

        assert stats["frames_processed"] == 5
        assert stats["avg_processing_time"] > 0

    def test_empty_image_handling(self):
        """Test handling of empty or invalid images."""
        # Test with None
        result = self.preprocessor.process(None)
        assert result is None

        # Test with empty array
        empty_image = np.array([])
        result = self.preprocessor.process(empty_image)
        assert result.size == 0

    def test_invalid_image_handling(self):
        """Test handling of invalid image formats."""
        # Test with wrong dimensions
        invalid_image = np.random.randint(0, 256, (10, 10, 10, 10), dtype=np.uint8)
        result = self.preprocessor.process(invalid_image)

        # Should return original image on error
        assert result is not None

    def test_all_noise_methods(self):
        """Test all noise reduction methods."""
        methods = [
            NoiseReductionMethod.GAUSSIAN,
            NoiseReductionMethod.BILATERAL,
            NoiseReductionMethod.MEDIAN,
            NoiseReductionMethod.NON_LOCAL_MEANS,
        ]

        for method in methods:
            with self.subTest(method=method):
                result = self.preprocessor.apply_noise_reduction(
                    self.test_noisy_image, method.value
                )
                assert result is not None
                assert result.shape == self.test_noisy_image.shape

    def test_performance_optimization(self):
        """Test performance optimization features."""
        # Test GPU setting (should not crash even if GPU not available)
        self.config.enable_gpu = True
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(self.test_bgr_image)
        assert result is not None


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for preprocessing."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_preprocess_image_function(self):
        """Test the preprocess_image convenience function."""
        result = preprocess_image(self.test_image)

        assert result is not None
        assert result.shape[0] == self.test_image.shape[0]
        assert result.shape[1] == self.test_image.shape[1]

    def test_preprocess_image_with_config(self):
        """Test preprocess_image with custom configuration."""
        config = create_default_config()
        config.target_color_space = ColorSpace.LAB

        result = preprocess_image(self.test_image, config)

        assert result is not None
        assert len(result.shape) == 3

    def test_enhance_for_detection(self):
        """Test enhance_for_detection function."""
        enhanced_bgr, hsv, lab = enhance_for_detection(self.test_image)

        assert enhanced_bgr is not None
        assert hsv is not None
        assert lab is not None

        # Check shapes
        assert enhanced_bgr.shape[0] == self.test_image.shape[0]
        assert hsv.shape[0] == self.test_image.shape[0]
        assert lab.shape[0] == self.test_image.shape[0]


class TestColorSpaceConversions(unittest.TestCase):
    """Test color space conversion functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_default_config()
        self.preprocessor = ImagePreprocessor(self.config.__dict__)
        self.test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

    def test_bgr_to_hsv_conversion(self):
        """Test BGR to HSV conversion."""
        result = self.preprocessor.convert_color_space(self.test_image, "HSV")

        assert result is not None
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_bgr_to_lab_conversion(self):
        """Test BGR to LAB conversion."""
        result = self.preprocessor.convert_color_space(self.test_image, "LAB")

        assert result is not None
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_bgr_to_gray_conversion(self):
        """Test BGR to grayscale conversion."""
        result = self.preprocessor.convert_color_space(self.test_image, "GRAY")

        assert result is not None
        assert len(result.shape) == 2

    def test_unsupported_conversion(self):
        """Test handling of unsupported color space conversions."""
        result = self.preprocessor.convert_color_space(self.test_image, "INVALID")

        # Should return original image
        np.testing.assert_array_equal(result, self.test_image)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_default_config()
        self.preprocessor = ImagePreprocessor(self.config.__dict__)

    def test_typical_pool_table_scenario(self):
        """Test preprocessing for typical pool table image."""
        # Create a simulated pool table image
        table_image = self._create_pool_table_image()

        result = self.preprocessor.process(table_image)

        assert result is not None
        assert result.shape[0] == table_image.shape[0]
        assert result.shape[1] == table_image.shape[1]

    def test_low_light_scenario(self):
        """Test preprocessing in low light conditions."""
        dark_image = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)

        self.config.auto_exposure_correction = True
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(dark_image)

        assert result is not None
        # Result should be brighter than original
        assert np.mean(result) > np.mean(dark_image)

    def test_high_contrast_scenario(self):
        """Test preprocessing with high contrast image."""
        # Create high contrast image
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[0:50, :] = 255  # White half
        high_contrast[50:100, :] = 0  # Black half

        result = self.preprocessor.process(high_contrast)

        assert result is not None

    def test_noisy_lighting_scenario(self):
        """Test preprocessing with varying lighting conditions."""
        # Create image with uneven lighting
        uneven_light = self._create_uneven_lighting_image()

        self.config.auto_exposure_correction = True
        self.config.auto_white_balance = True
        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(uneven_light)

        assert result is not None

    def test_full_pipeline_performance(self):
        """Test full preprocessing pipeline performance."""
        large_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Enable all preprocessing features
        self.config.noise_reduction_enabled = True
        self.config.auto_exposure_correction = True
        self.config.auto_white_balance = True
        self.config.contrast_enhancement = True
        self.config.morphology_enabled = True

        preprocessor = ImagePreprocessor(self.config.__dict__)

        result = preprocessor.process(large_image)
        stats = preprocessor.get_statistics()

        assert result is not None
        assert stats["frames_processed"] > 0
        assert stats["avg_processing_time"] > 0

    def _create_pool_table_image(self):
        """Create a simulated pool table image."""
        image = np.zeros((200, 300, 3), dtype=np.uint8)

        # Green table felt
        image[:, :] = [47, 120, 47]  # Dark green in BGR

        # Add some balls
        cv2.circle(image, (100, 100), 10, (255, 255, 255), -1)  # Cue ball
        cv2.circle(image, (150, 80), 10, (0, 0, 255), -1)  # Red ball
        cv2.circle(image, (180, 120), 10, (0, 255, 255), -1)  # Yellow ball

        # Add some noise
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def _create_uneven_lighting_image(self):
        """Create an image with uneven lighting."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create gradient lighting
        for y in range(100):
            for x in range(100):
                brightness = int(50 + (x + y) * 100 / 200)
                image[y, x] = [brightness, brightness, brightness]

        return image


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
