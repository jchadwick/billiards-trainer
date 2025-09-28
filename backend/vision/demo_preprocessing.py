#!/usr/bin/env python3
"""Image Preprocessing Demo.

Demonstrates the image preprocessing pipeline with before/after comparisons.
Shows the effects of each preprocessing step on sample billiards images.

Usage:
    python demo_preprocessing.py [--input <image_path>] [--output <dir>] [--save-steps]

Features:
- Loads sample billiards images or uses provided input
- Applies preprocessing pipeline step by step
- Shows before/after comparisons
- Generates visual output with side-by-side comparisons
- Saves intermediate processing steps if requested
- Displays performance metrics
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .preprocessing import (
    ColorSpace,
    ImagePreprocessor,
    NoiseReductionMethod,
    PreprocessingConfig,
)


class PreprocessingDemo:
    """Demo class for showcasing image preprocessing capabilities."""

    def __init__(
        self, save_output: bool = True, output_dir: str = "preprocessing_demo_output"
    ):
        """Initialize demo with configuration.

        Args:
            save_output: Whether to save output images
            output_dir: Directory to save output images
        """
        self.save_output = save_output
        self.output_dir = Path(output_dir)

        if self.save_output:
            self.output_dir.mkdir(exist_ok=True)
            print(f"Output will be saved to: {self.output_dir.absolute()}")

    def create_sample_images(self) -> dict[str, np.ndarray]:
        """Create sample test images for demonstration.

        Returns:
            Dictionary of sample images
        """
        samples = {}

        # 1. Synthetic pool table image
        samples["synthetic_table"] = self._create_synthetic_table()

        # 2. Noisy image
        samples["noisy_image"] = self._create_noisy_image()

        # 3. Dark image (poor lighting)
        samples["dark_image"] = self._create_dark_image()

        # 4. Color cast image (poor white balance)
        samples["color_cast"] = self._create_color_cast_image()

        # 5. Uneven lighting
        samples["uneven_lighting"] = self._create_uneven_lighting()

        return samples

    def _create_synthetic_table(self) -> np.ndarray:
        """Create a synthetic pool table image."""
        # Create green table background
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        image[:, :] = [34, 139, 34]  # Forest green in BGR

        # Add table rails (darker edges)
        cv2.rectangle(image, (0, 0), (600, 50), (20, 80, 20), -1)
        cv2.rectangle(image, (0, 350), (600, 400), (20, 80, 20), -1)
        cv2.rectangle(image, (0, 0), (50, 400), (20, 80, 20), -1)
        cv2.rectangle(image, (550, 0), (600, 400), (20, 80, 20), -1)

        # Add pockets (black circles at corners and sides)
        pockets = [(30, 30), (300, 30), (570, 30), (30, 370), (300, 370), (570, 370)]
        for pocket in pockets:
            cv2.circle(image, pocket, 25, (0, 0, 0), -1)

        # Add balls
        balls = [
            ((150, 200), (255, 255, 255)),  # Cue ball (white)
            ((300, 180), (0, 0, 255)),  # Red ball
            ((320, 200), (0, 255, 255)),  # Yellow ball
            ((280, 220), (255, 0, 0)),  # Blue ball
            ((340, 180), (0, 165, 255)),  # Orange ball
            ((290, 160), (128, 0, 128)),  # Purple ball
            ((310, 240), (0, 128, 0)),  # Green ball
            ((330, 160), (139, 69, 19)),  # Brown ball
            ((270, 200), (0, 0, 0)),  # 8-ball (black)
        ]

        for (x, y), color in balls:
            cv2.circle(image, (x, y), 12, color, -1)
            cv2.circle(image, (x, y), 12, (255, 255, 255), 1)  # White outline

        # Add some noise for realism
        noise = np.random.normal(0, 8, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def _create_noisy_image(self) -> np.ndarray:
        """Create a noisy version of the synthetic table."""
        base = self._create_synthetic_table()

        # Add significant noise
        noise = np.random.normal(0, 25, base.shape).astype(np.int16)
        noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add salt and pepper noise
        sp_noise = np.random.random(base.shape[:2])
        noisy[sp_noise < 0.01] = 0  # Salt
        noisy[sp_noise > 0.99] = 255  # Pepper

        return noisy

    def _create_dark_image(self) -> np.ndarray:
        """Create a dark image simulating poor lighting."""
        base = self._create_synthetic_table()

        # Darken the image significantly
        dark = (base.astype(np.float32) * 0.3).astype(np.uint8)

        # Add uneven lighting (darker on the right)
        height, width = dark.shape[:2]
        for x in range(width):
            factor = 1.0 - (x / width) * 0.5
            dark[:, x] = np.clip(dark[:, x].astype(np.float32) * factor, 0, 255).astype(
                np.uint8
            )

        return dark

    def _create_color_cast_image(self) -> np.ndarray:
        """Create an image with color cast (poor white balance)."""
        base = self._create_synthetic_table()

        # Add blue color cast
        cast = base.copy().astype(np.float32)
        cast[:, :, 0] *= 1.4  # Increase blue channel
        cast[:, :, 1] *= 0.9  # Decrease green channel
        cast[:, :, 2] *= 0.8  # Decrease red channel

        return np.clip(cast, 0, 255).astype(np.uint8)

    def _create_uneven_lighting(self) -> np.ndarray:
        """Create an image with uneven lighting."""
        base = self._create_synthetic_table()
        height, width = base.shape[:2]

        # Create lighting gradient
        uneven = base.copy().astype(np.float32)

        for y in range(height):
            for x in range(width):
                # Create circular lighting pattern
                center_x, center_y = width // 3, height // 2
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                max_distance = np.sqrt(width**2 + height**2) / 2

                # Lighting factor: brighter in center, darker at edges
                lighting_factor = 0.4 + 0.6 * (1 - min(distance / max_distance, 1))
                uneven[y, x] *= lighting_factor

        return np.clip(uneven, 0, 255).astype(np.uint8)

    def demonstrate_color_space_conversions(
        self, image: np.ndarray, name: str = "image"
    ):
        """Demonstrate different color space conversions."""
        print(f"\n=== Color Space Conversions for {name} ===")

        config = PreprocessingConfig()
        preprocessor = ImagePreprocessor(config.__dict__)

        # Convert to different color spaces
        conversions = {
            "HSV": ColorSpace.HSV,
            "LAB": ColorSpace.LAB,
            "RGB": ColorSpace.RGB,
            "Grayscale": ColorSpace.GRAY,
        }

        results = {"Original (BGR)": image}

        for space_name, _color_space in conversions.items():
            try:
                converted = preprocessor.convert_color_space(image, space_name)
                results[space_name] = converted
                print(f"✓ {space_name}: {converted.shape}")
            except Exception as e:
                print(f"✗ {space_name}: Error - {e}")

        if self.save_output:
            self._save_comparison_grid(results, f"{name}_color_spaces")

        return results

    def demonstrate_noise_reduction(self, noisy_image: np.ndarray, name: str = "noisy"):
        """Demonstrate different noise reduction methods."""
        print(f"\n=== Noise Reduction Methods for {name} ===")

        config = PreprocessingConfig()
        config.noise_reduction_enabled = True

        methods = {
            "Original": None,
            "Gaussian": NoiseReductionMethod.GAUSSIAN,
            "Bilateral": NoiseReductionMethod.BILATERAL,
            "Median": NoiseReductionMethod.MEDIAN,
            "Non-Local Means": NoiseReductionMethod.NON_LOCAL_MEANS,
        }

        results = {}

        for method_name, method in methods.items():
            try:
                if method is None:
                    results[method_name] = noisy_image
                else:
                    config.noise_method = method
                    preprocessor = ImagePreprocessor(config.__dict__)
                    denoised = preprocessor.apply_noise_reduction(
                        noisy_image, method.value
                    )
                    results[method_name] = denoised

                    # Calculate noise reduction effectiveness
                    original_std = np.std(noisy_image)
                    denoised_std = np.std(denoised)
                    reduction = (original_std - denoised_std) / original_std * 100
                    print(f"✓ {method_name}: {reduction:.1f}% noise reduction")

            except Exception as e:
                print(f"✗ {method_name}: Error - {e}")

        if self.save_output:
            self._save_comparison_grid(results, f"{name}_noise_reduction")

        return results

    def demonstrate_exposure_correction(
        self, dark_image: np.ndarray, name: str = "dark"
    ):
        """Demonstrate exposure and brightness correction."""
        print(f"\n=== Exposure Correction for {name} ===")

        config = PreprocessingConfig()

        # Different exposure correction settings
        settings = {
            "Original": {
                "auto_exposure_correction": False,
                "contrast_enhancement": False,
            },
            "Auto Exposure": {
                "auto_exposure_correction": True,
                "contrast_enhancement": False,
            },
            "Contrast Enhancement": {
                "auto_exposure_correction": False,
                "contrast_enhancement": True,
            },
            "Both": {"auto_exposure_correction": True, "contrast_enhancement": True},
        }

        results = {}

        for setting_name, setting in settings.items():
            try:
                config.auto_exposure_correction = setting["auto_exposure_correction"]
                config.contrast_enhancement = setting["contrast_enhancement"]

                preprocessor = ImagePreprocessor(config.__dict__)

                if setting_name == "Original":
                    results[setting_name] = dark_image
                else:
                    corrected = preprocessor.process(dark_image)
                    results[setting_name] = corrected

                    # Calculate brightness improvement
                    original_brightness = np.mean(
                        cv2.cvtColor(dark_image, cv2.COLOR_BGR2GRAY)
                    )
                    corrected_brightness = np.mean(
                        cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
                    )
                    improvement = corrected_brightness - original_brightness
                    print(f"✓ {setting_name}: +{improvement:.1f} brightness increase")

            except Exception as e:
                print(f"✗ {setting_name}: Error - {e}")

        if self.save_output:
            self._save_comparison_grid(results, f"{name}_exposure_correction")

        return results

    def demonstrate_white_balance(
        self, color_cast_image: np.ndarray, name: str = "color_cast"
    ):
        """Demonstrate white balance correction."""
        print(f"\n=== White Balance Correction for {name} ===")

        config = PreprocessingConfig()

        results = {"Original": color_cast_image}

        # Test white balance correction
        try:
            config.auto_white_balance = True
            preprocessor = ImagePreprocessor(config.__dict__)

            corrected = preprocessor.process(color_cast_image)
            results["White Balance Corrected"] = corrected

            # Calculate color balance improvement
            original_means = np.mean(color_cast_image, axis=(0, 1))
            corrected_means = np.mean(corrected, axis=(0, 1))

            print(f"✓ Original BGR means: {original_means}")
            print(f"✓ Corrected BGR means: {corrected_means}")

        except Exception as e:
            print(f"✗ White Balance: Error - {e}")

        if self.save_output:
            self._save_comparison_grid(results, f"{name}_white_balance")

        return results

    def demonstrate_full_pipeline(self, image: np.ndarray, name: str = "image"):
        """Demonstrate the complete preprocessing pipeline."""
        print(f"\n=== Full Preprocessing Pipeline for {name} ===")

        # Create configuration for optimal detection
        config = PreprocessingConfig()
        config.debug_mode = True
        config.save_intermediate_steps = True

        preprocessor = ImagePreprocessor(config.__dict__)

        # Time the processing
        start_time = time.time()
        result = preprocessor.process(image)
        processing_time = (time.time() - start_time) * 1000

        print(f"✓ Processing completed in {processing_time:.2f}ms")

        # Get debug images if available
        debug_images = preprocessor.get_debug_images()

        results = {"Original": image, "Final Result": result}

        # Add debug steps if available
        for step_name, step_image in debug_images:
            results[f"Step: {step_name}"] = step_image

        # Get statistics
        stats = preprocessor.get_statistics()
        print(f"✓ Statistics: {stats}")

        if self.save_output:
            self._save_comparison_grid(results, f"{name}_full_pipeline")

        return result, stats

    def demonstrate_detection_optimization(
        self, image: np.ndarray, name: str = "image"
    ):
        """Demonstrate optimization specifically for detection algorithms."""
        print(f"\n=== Detection Optimization for {name} ===")

        try:
            # Create optimized config for detection
            config = PreprocessingConfig(
                target_color_space=ColorSpace.HSV,
                noise_reduction_enabled=True,
                noise_method=NoiseReductionMethod.BILATERAL,
                auto_exposure_correction=True,
                auto_white_balance=True,
                morphology_enabled=True,
            )

            preprocessor = ImagePreprocessor(config.__dict__)
            enhanced_bgr = preprocessor.process(image)

            # Convert to different color spaces
            hsv = preprocessor.convert_color_space(image, "HSV")
            lab = preprocessor.convert_color_space(image, "LAB")

            results = {
                "Original BGR": image,
                "Enhanced BGR": enhanced_bgr,
                "HSV": hsv,
                "LAB": lab,
            }

            # Analyze each color space for detection suitability
            for space_name, space_image in results.items():
                if space_name == "Original BGR":
                    continue

                # Calculate contrast and other metrics
                if len(space_image.shape) == 3:
                    gray = cv2.cvtColor(space_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = space_image

                contrast = np.std(gray)
                brightness = np.mean(gray)

                print(
                    f"✓ {space_name}: Contrast={contrast:.1f}, Brightness={brightness:.1f}"
                )

            if self.save_output:
                self._save_comparison_grid(results, f"{name}_detection_optimization")

            return results

        except Exception as e:
            print(f"✗ Detection Optimization: Error - {e}")
            return {}

    def _save_comparison_grid(self, images: dict[str, np.ndarray], filename: str):
        """Save a grid comparison of images."""
        if not images:
            return

        num_images = len(images)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        fig.suptitle(filename.replace("_", " ").title(), fontsize=16)

        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten() if num_images > 1 else [axes]

        for idx, (name, image) in enumerate(images.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Convert BGR to RGB for matplotlib if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image

            ax.imshow(display_image, cmap="gray" if len(image.shape) == 2 else None)
            ax.set_title(name, fontsize=10)
            ax.axis("off")

        # Hide unused subplots
        for idx in range(num_images, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved comparison: {output_path}")

    def run_full_demo(self, input_image: np.ndarray = None, image_name: str = "input"):
        """Run the complete preprocessing demonstration."""
        print("🎱 Billiards Vision Preprocessing Demo")
        print("=" * 50)

        if input_image is not None:
            # Use provided image
            test_images = {image_name: input_image}
        else:
            # Create sample images
            print("Creating sample test images...")
            test_images = self.create_sample_images()

        for name, image in test_images.items():
            print(f"\n🖼️  Processing: {name}")
            print("-" * 30)

            # Demonstrate each preprocessing aspect
            self.demonstrate_color_space_conversions(image, name)

            if "noisy" in name.lower():
                self.demonstrate_noise_reduction(image, name)

            if "dark" in name.lower():
                self.demonstrate_exposure_correction(image, name)

            if "color_cast" in name.lower() or "cast" in name.lower():
                self.demonstrate_white_balance(image, name)

            # Always demonstrate full pipeline and detection optimization
            self.demonstrate_full_pipeline(image, name)
            self.demonstrate_detection_optimization(image, name)

        print(f"\n✅ Demo completed! Check {self.output_dir} for output images.")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Image Preprocessing Demo")
    parser.add_argument("--input", "-i", type=str, help="Input image path")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="preprocessing_demo_output",
        help="Output directory",
    )
    parser.add_argument(
        "--save-steps", action="store_true", help="Save intermediate processing steps"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Do not display images (save only)"
    )

    args = parser.parse_args()

    # Create demo instance
    demo = PreprocessingDemo(save_output=True, output_dir=args.output)

    # Load input image if provided
    input_image = None
    image_name = "input"

    if args.input:
        if os.path.exists(args.input):
            input_image = cv2.imread(args.input)
            if input_image is not None:
                image_name = Path(args.input).stem
                print(f"Loaded input image: {args.input}")
            else:
                print(f"Error: Could not load image from {args.input}")
                return 1
        else:
            print(f"Error: Input file {args.input} does not exist")
            return 1

    try:
        # Run the demo
        demo.run_full_demo(input_image, image_name)

        print("\n📊 Performance Summary:")
        print(f"   Output directory: {demo.output_dir.absolute()}")

        if not args.no_display:
            print("   Open the output directory to view generated comparisons")

        return 0

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
