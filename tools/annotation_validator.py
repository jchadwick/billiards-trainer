#!/usr/bin/env python3
"""
YOLO Dataset Annotation Validator

This tool validates YOLO format annotations and generates comprehensive dataset statistics.
It checks for common issues like missing labels, invalid formats, out-of-range values,
and provides visualizations of annotated samples.

Usage:
    python tools/annotation_validator.py [dataset_path]
    python tools/annotation_validator.py --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    import cv2
except ImportError:
    print("ERROR: Required packages not installed!")
    print("\nPlease install dependencies:")
    print("  pip install pillow opencv-python numpy")
    sys.exit(1)


@dataclass
class ValidationError:
    """Represents a validation error"""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'missing_label', 'invalid_format', 'out_of_range', etc.
    file_path: str
    line_number: Optional[int] = None
    message: str = ""
    details: Optional[Dict[str, Any]] = None


@dataclass
class DatasetStats:
    """Dataset statistics"""
    total_images: int = 0
    total_labels: int = 0
    images_with_labels: int = 0
    images_without_labels: int = 0
    total_annotations: int = 0
    class_distribution: Dict[int, int] = None
    bbox_sizes: List[float] = None  # areas as percentages
    image_dimensions: Dict[str, List[int]] = None  # width, height lists
    empty_label_files: int = 0

    def __post_init__(self):
        if self.class_distribution is None:
            self.class_distribution = {}
        if self.bbox_sizes is None:
            self.bbox_sizes = []
        if self.image_dimensions is None:
            self.image_dimensions = {'widths': [], 'heights': []}


class AnnotationValidator:
    """Validates YOLO format dataset annotations"""

    # Standard YOLO image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # Color palette for visualization (BGR format for OpenCV)
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255), (192, 192, 192)
    ]

    def __init__(self, dataset_path: Path, num_classes: int = 19):
        """
        Initialize validator

        Args:
            dataset_path: Path to dataset root directory
            num_classes: Number of valid classes (default: 19 for billiards dataset)
        """
        self.dataset_path = Path(dataset_path)
        self.num_classes = num_classes
        self.errors: List[ValidationError] = []
        self.stats = DatasetStats()

        # Load class names if available
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> Dict[int, str]:
        """Load class names from dataset.yaml or classes.txt"""
        class_names = {}

        # Try dataset.yaml first
        yaml_path = self.dataset_path / "dataset.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('#') or not line:
                            continue
                        # Parse "  0: cue_ball" format
                        if ':' in line and line[0].isspace():
                            parts = line.split(':', 1)
                            try:
                                class_id = int(parts[0].strip())
                                class_name = parts[1].split('#')[0].strip()
                                class_names[class_id] = class_name
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                print(f"Warning: Could not parse dataset.yaml: {e}")

        # Try classes.txt as fallback
        if not class_names:
            classes_path = self.dataset_path / "classes.txt"
            if classes_path.exists():
                try:
                    with open(classes_path) as f:
                        for idx, line in enumerate(f):
                            class_name = line.strip()
                            if class_name:
                                class_names[idx] = class_name
                except Exception as e:
                    print(f"Warning: Could not read classes.txt: {e}")

        # Generate default names if none found
        if not class_names:
            class_names = {i: f"class_{i}" for i in range(self.num_classes)}
            print(f"Warning: No class names found, using defaults")

        return class_names

    def validate(self, splits: List[str] = None) -> Tuple[bool, DatasetStats]:
        """
        Validate the dataset

        Args:
            splits: List of splits to validate (e.g., ['train', 'val', 'test'])
                   If None, validates all found splits

        Returns:
            Tuple of (is_valid, stats)
        """
        print("=" * 80)
        print("YOLO Dataset Annotation Validator")
        print("=" * 80)
        print(f"\nDataset path: {self.dataset_path}")
        print(f"Valid classes: 0-{self.num_classes - 1}")

        # Determine which splits to validate
        if splits is None:
            splits = self._find_splits()

        if not splits:
            self._add_error('error', 'structure', str(self.dataset_path),
                          message="No image directories found (train/val/test)")
            return False, self.stats

        print(f"Validating splits: {', '.join(splits)}\n")

        # Validate each split
        for split in splits:
            print(f"\n{'=' * 80}")
            print(f"Validating: {split}")
            print('=' * 80)
            self._validate_split(split)

        # Print summary
        self._print_summary()

        # Determine if validation passed
        has_errors = any(e.severity == 'error' for e in self.errors)
        return not has_errors, self.stats

    def _find_splits(self) -> List[str]:
        """Find available splits in the dataset"""
        splits = []
        images_dir = self.dataset_path / "images"

        if not images_dir.exists():
            return splits

        for split_dir in images_dir.iterdir():
            if split_dir.is_dir() and split_dir.name in ['train', 'val', 'test']:
                splits.append(split_dir.name)

        return sorted(splits)

    def _validate_split(self, split: str):
        """Validate a single split (train/val/test)"""
        images_dir = self.dataset_path / "images" / split
        labels_dir = self.dataset_path / "labels" / split

        # Check if directories exist
        if not images_dir.exists():
            self._add_error('error', 'structure', str(images_dir),
                          message=f"Images directory not found for split '{split}'")
            return

        if not labels_dir.exists():
            self._add_error('error', 'structure', str(labels_dir),
                          message=f"Labels directory not found for split '{split}'")
            return

        # Get all image files
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))

        if not image_files:
            self._add_error('warning', 'structure', str(images_dir),
                          message=f"No images found in split '{split}'")
            return

        print(f"Found {len(image_files)} images")

        # Validate each image and its label
        split_images_with_labels = 0
        split_images_without_labels = 0

        for image_path in image_files:
            self.stats.total_images += 1

            # Check for corresponding label file
            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                self._add_error('warning', 'missing_label', str(image_path),
                              message="No corresponding label file")
                split_images_without_labels += 1
                continue

            self.stats.total_labels += 1

            # Validate label file
            has_annotations = self._validate_label_file(label_path, image_path)

            if has_annotations:
                split_images_with_labels += 1
            else:
                split_images_without_labels += 1

        # Update stats
        self.stats.images_with_labels += split_images_with_labels
        self.stats.images_without_labels += split_images_without_labels

        print(f"\n{split} Summary:")
        print(f"  Images: {len(image_files)}")
        print(f"  Images with annotations: {split_images_with_labels}")
        print(f"  Images without annotations: {split_images_without_labels}")

    def _validate_label_file(self, label_path: Path, image_path: Path) -> bool:
        """
        Validate a single label file

        Returns:
            True if file has valid annotations, False if empty or all invalid
        """
        try:
            with open(label_path) as f:
                lines = f.readlines()
        except Exception as e:
            self._add_error('error', 'read_error', str(label_path),
                          message=f"Could not read file: {e}")
            return False

        # Check if file is empty
        lines = [line.strip() for line in lines if line.strip()]
        if not lines:
            self.stats.empty_label_files += 1
            self._add_error('info', 'empty_label', str(label_path),
                          message="Label file is empty")
            return False

        # Get image dimensions for validation
        image_width, image_height = self._get_image_dimensions(image_path)
        if image_width and image_height:
            self.stats.image_dimensions['widths'].append(image_width)
            self.stats.image_dimensions['heights'].append(image_height)

        # Validate each annotation line
        valid_annotations = 0
        for line_num, line in enumerate(lines, start=1):
            if self._validate_annotation_line(line, line_num, label_path):
                valid_annotations += 1
                self.stats.total_annotations += 1

        return valid_annotations > 0

    def _validate_annotation_line(self, line: str, line_num: int, label_path: Path) -> bool:
        """
        Validate a single annotation line

        YOLO format: class_id x_center y_center width height
        All values normalized to [0, 1]

        Returns:
            True if annotation is valid
        """
        parts = line.split()

        # Check format: must have exactly 5 values
        if len(parts) != 5:
            self._add_error('error', 'invalid_format', str(label_path), line_num,
                          message=f"Expected 5 values, got {len(parts)}: {line}",
                          details={'line': line, 'num_parts': len(parts)})
            return False

        try:
            # Parse values
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Validate class ID
            if not (0 <= class_id < self.num_classes):
                self._add_error('error', 'invalid_class', str(label_path), line_num,
                              message=f"Class ID {class_id} out of range [0, {self.num_classes - 1}]",
                              details={'class_id': class_id, 'line': line})
                return False

            # Validate bounding box coordinates (should be in [0, 1])
            bbox_errors = []
            if not (0 <= x_center <= 1):
                bbox_errors.append(f"x_center={x_center}")
            if not (0 <= y_center <= 1):
                bbox_errors.append(f"y_center={y_center}")
            if not (0 < width <= 1):
                bbox_errors.append(f"width={width}")
            if not (0 < height <= 1):
                bbox_errors.append(f"height={height}")

            if bbox_errors:
                self._add_error('error', 'out_of_range', str(label_path), line_num,
                              message=f"Bbox values out of range [0, 1]: {', '.join(bbox_errors)}",
                              details={'line': line, 'issues': bbox_errors})
                return False

            # Check for suspicious values
            if width < 0.001 or height < 0.001:
                self._add_error('warning', 'tiny_bbox', str(label_path), line_num,
                              message=f"Unusually small bbox: {width}x{height}",
                              details={'line': line, 'width': width, 'height': height})

            if width > 0.9 or height > 0.9:
                self._add_error('warning', 'large_bbox', str(label_path), line_num,
                              message=f"Unusually large bbox: {width}x{height}",
                              details={'line': line, 'width': width, 'height': height})

            # Update statistics
            if class_id not in self.stats.class_distribution:
                self.stats.class_distribution[class_id] = 0
            self.stats.class_distribution[class_id] += 1

            # Store bbox size (area as percentage)
            bbox_area = width * height
            self.stats.bbox_sizes.append(bbox_area)

            return True

        except ValueError as e:
            self._add_error('error', 'invalid_format', str(label_path), line_num,
                          message=f"Could not parse values: {e}",
                          details={'line': line, 'error': str(e)})
            return False

    def _get_image_dimensions(self, image_path: Path) -> Tuple[Optional[int], Optional[int]]:
        """Get image dimensions (width, height)"""
        try:
            with Image.open(image_path) as img:
                return img.width, img.height
        except Exception as e:
            self._add_error('warning', 'image_read_error', str(image_path),
                          message=f"Could not read image dimensions: {e}")
            return None, None

    def _add_error(self, severity: str, category: str, file_path: str,
                   line_number: Optional[int] = None, message: str = "",
                   details: Optional[Dict[str, Any]] = None):
        """Add a validation error"""
        error = ValidationError(
            severity=severity,
            category=category,
            file_path=file_path,
            line_number=line_number,
            message=message,
            details=details
        )
        self.errors.append(error)

    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        # Count errors by severity
        errors_count = sum(1 for e in self.errors if e.severity == 'error')
        warnings_count = sum(1 for e in self.errors if e.severity == 'warning')
        info_count = sum(1 for e in self.errors if e.severity == 'info')

        print(f"\nTotal Issues: {len(self.errors)}")
        print(f"  Errors:   {errors_count}")
        print(f"  Warnings: {warnings_count}")
        print(f"  Info:     {info_count}")

        # Print errors by category
        if self.errors:
            print("\nIssues by Category:")
            category_counts = defaultdict(int)
            for error in self.errors:
                category_counts[error.category] += 1

            for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                print(f"  {category}: {count}")

            # Print first few errors of each type
            print("\nSample Issues:")
            printed_categories = set()
            for error in self.errors[:20]:  # Limit to first 20 errors
                if error.category not in printed_categories:
                    printed_categories.add(error.category)
                    location = f"{error.file_path}"
                    if error.line_number:
                        location += f":{error.line_number}"
                    print(f"\n  [{error.severity.upper()}] {error.category}")
                    print(f"    Location: {location}")
                    print(f"    {error.message}")

                if len(printed_categories) >= 5:  # Show at most 5 different categories
                    break

            if len(self.errors) > 20:
                print(f"\n  ... and {len(self.errors) - 20} more issues")

        # Print dataset statistics
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)

        print(f"\nImages:")
        print(f"  Total images: {self.stats.total_images}")
        print(f"  Images with labels: {self.stats.total_labels}")
        print(f"  Images with annotations: {self.stats.images_with_labels}")
        print(f"  Images without annotations: {self.stats.images_without_labels}")
        print(f"  Empty label files: {self.stats.empty_label_files}")

        print(f"\nAnnotations:")
        print(f"  Total annotations: {self.stats.total_annotations}")
        if self.stats.total_annotations > 0:
            print(f"  Average annotations per image: {self.stats.total_annotations / max(1, self.stats.images_with_labels):.2f}")

        # Class distribution
        if self.stats.class_distribution:
            print("\nClass Distribution:")
            sorted_classes = sorted(self.stats.class_distribution.items())
            for class_id, count in sorted_classes:
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                percentage = (count / self.stats.total_annotations) * 100
                bar_length = int(percentage / 2)  # Scale to fit in terminal
                bar = "█" * bar_length
                print(f"  {class_id:2d} {class_name:20s}: {count:5d} ({percentage:5.1f}%) {bar}")

        # Bounding box statistics
        if self.stats.bbox_sizes:
            bbox_areas = np.array(self.stats.bbox_sizes)
            print("\nBounding Box Sizes (as % of image area):")
            print(f"  Mean:   {bbox_areas.mean() * 100:.2f}%")
            print(f"  Median: {np.median(bbox_areas) * 100:.2f}%")
            print(f"  Min:    {bbox_areas.min() * 100:.2f}%")
            print(f"  Max:    {bbox_areas.max() * 100:.2f}%")
            print(f"  Std:    {bbox_areas.std() * 100:.2f}%")

        # Image dimensions
        if self.stats.image_dimensions['widths']:
            widths = np.array(self.stats.image_dimensions['widths'])
            heights = np.array(self.stats.image_dimensions['heights'])
            print("\nImage Dimensions:")
            print(f"  Width:  {widths.min()}x{widths.max()} (avg: {widths.mean():.0f})")
            print(f"  Height: {heights.min()}x{heights.max()} (avg: {heights.mean():.0f})")

            # Check for consistency
            unique_dims = set(zip(widths, heights))
            if len(unique_dims) == 1:
                print("  ✓ All images have the same dimensions")
            else:
                print(f"  ⚠ Found {len(unique_dims)} different image dimensions")

        # Final verdict
        print("\n" + "=" * 80)
        if errors_count == 0:
            print("✅ VALIDATION PASSED")
            if warnings_count > 0:
                print(f"   Note: {warnings_count} warnings found (not critical)")
        else:
            print("❌ VALIDATION FAILED")
            print(f"   Found {errors_count} errors that must be fixed")
        print("=" * 80 + "\n")

    def save_stats(self, output_path: Path):
        """Save statistics to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert stats to dict
        stats_dict = asdict(self.stats)

        # Add metadata
        stats_dict['metadata'] = {
            'dataset_path': str(self.dataset_path),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
        }

        # Add numpy arrays as lists
        if self.stats.bbox_sizes:
            stats_dict['bbox_size_percentiles'] = {
                'p10': float(np.percentile(self.stats.bbox_sizes, 10) * 100),
                'p25': float(np.percentile(self.stats.bbox_sizes, 25) * 100),
                'p50': float(np.percentile(self.stats.bbox_sizes, 50) * 100),
                'p75': float(np.percentile(self.stats.bbox_sizes, 75) * 100),
                'p90': float(np.percentile(self.stats.bbox_sizes, 90) * 100),
            }

        # Don't save the full bbox_sizes list (too large)
        stats_dict['bbox_sizes'] = None

        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

        print(f"Statistics saved to: {output_path}")

    def visualize_samples(self, output_dir: Path, num_samples: int = 10, splits: List[str] = None):
        """
        Visualize sample annotations

        Args:
            output_dir: Directory to save visualization images
            num_samples: Number of samples to visualize per split
            splits: List of splits to visualize (default: all found splits)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if splits is None:
            splits = self._find_splits()

        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        for split in splits:
            print(f"\nGenerating {num_samples} samples for {split} split...")
            self._visualize_split_samples(split, output_dir, num_samples)

        print(f"\n✓ Visualizations saved to: {output_dir}")

    def _visualize_split_samples(self, split: str, output_dir: Path, num_samples: int):
        """Visualize samples from a specific split"""
        images_dir = self.dataset_path / "images" / split
        labels_dir = self.dataset_path / "labels" / split

        if not images_dir.exists() or not labels_dir.exists():
            print(f"  ⚠ Skip {split}: directories not found")
            return

        # Get all images with labels
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(images_dir.glob(f"*{ext}"))

        # Filter to only images with non-empty labels
        images_with_labels = []
        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                try:
                    with open(label_path) as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if lines:
                            images_with_labels.append(image_path)
                except:
                    pass

        if not images_with_labels:
            print(f"  ⚠ No annotated images found in {split}")
            return

        # Sample random images
        np.random.seed(42)  # For reproducibility
        sample_size = min(num_samples, len(images_with_labels))
        sampled_images = np.random.choice(images_with_labels, sample_size, replace=False)

        # Visualize each sample
        for idx, image_path in enumerate(sampled_images):
            try:
                self._visualize_single_image(image_path, labels_dir, output_dir, split, idx)
            except Exception as e:
                print(f"  ⚠ Error visualizing {image_path.name}: {e}")

        print(f"  ✓ Generated {sample_size} visualizations")

    def _visualize_single_image(self, image_path: Path, labels_dir: Path,
                                 output_dir: Path, split: str, idx: int):
        """Visualize a single annotated image"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return

        height, width = image.shape[:2]

        # Read labels
        label_path = labels_dir / f"{image_path.stem}.txt"
        with open(label_path) as f:
            annotations = [line.strip().split() for line in f if line.strip()]

        # Draw annotations
        for annotation in annotations:
            try:
                class_id = int(annotation[0])
                x_center = float(annotation[1])
                y_center = float(annotation[2])
                bbox_width = float(annotation[3])
                bbox_height = float(annotation[4])

                # Convert YOLO format to pixel coordinates
                x_center_px = int(x_center * width)
                y_center_px = int(y_center * height)
                w_px = int(bbox_width * width)
                h_px = int(bbox_height * height)

                x1 = int(x_center_px - w_px / 2)
                y1 = int(y_center_px - h_px / 2)
                x2 = int(x_center_px + w_px / 2)
                y2 = int(y_center_px + h_px / 2)

                # Get color for this class
                color = self.COLORS[class_id % len(self.COLORS)]

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                label_text = f"{class_name} ({class_id})"

                # Calculate label background size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, font_thickness
                )

                # Draw label background
                cv2.rectangle(image, (x1, y1 - text_height - 10),
                            (x1 + text_width + 10, y1), color, -1)

                # Draw label text
                cv2.putText(image, label_text, (x1 + 5, y1 - 5),
                          font, font_scale, (255, 255, 255), font_thickness)

            except (ValueError, IndexError):
                continue

        # Add info text
        info_text = f"{split}/{image_path.name} - {len(annotations)} annotations"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Save image
        output_path = output_dir / f"{split}_sample_{idx:03d}_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), image)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate YOLO dataset annotations and generate statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate dataset in current directory
  python tools/annotation_validator.py

  # Validate specific dataset
  python tools/annotation_validator.py /path/to/dataset

  # Validate with custom number of classes
  python tools/annotation_validator.py --num-classes 80

  # Generate more visualization samples
  python tools/annotation_validator.py --num-samples 20

  # Validate specific splits only
  python tools/annotation_validator.py --splits train val
        """
    )

    parser.add_argument(
        'dataset_path',
        nargs='?',
        default='dataset',
        help='Path to dataset root directory (default: dataset)'
    )

    parser.add_argument(
        '--num-classes',
        type=int,
        default=19,
        help='Number of valid classes (default: 19 for billiards dataset)'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of visualization samples per split (default: 10)'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        choices=['train', 'val', 'test'],
        help='Specific splits to validate (default: all found splits)'
    )

    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip generating visualization samples'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for visualizations and stats (default: dataset/validation_samples)'
    )

    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path

    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return 1

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = dataset_path / "validation_samples"

    # Create validator
    validator = AnnotationValidator(dataset_path, num_classes=args.num_classes)

    # Run validation
    is_valid, stats = validator.validate(splits=args.splits)

    # Save statistics
    stats_path = output_dir / "dataset_stats.json"
    validator.save_stats(stats_path)

    # Generate visualizations
    if not args.no_visualize:
        validator.visualize_samples(output_dir, num_samples=args.num_samples, splits=args.splits)

    # Return appropriate exit code
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
