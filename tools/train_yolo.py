#!/usr/bin/env python3
"""
Interactive YOLO Training Script for Billiards Detection

This script provides a user-friendly, step-by-step interface for training a YOLOv8 model
on billiard ball and cue stick detection data. It's designed to be idiot-proof with
extensive validation, helpful prompts, and automatic error handling.

Usage:
    python tools/train_yolo.py [--auto]  # --auto skips interactive prompts
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("❌ ERROR: Required packages not installed!")
    print("\nPlease install training dependencies:")
    print("  pip install ultralytics torch torchvision")
    sys.exit(1)


class YOLOTrainer:
    """Interactive YOLO training wizard for billiards detection"""

    def __init__(self, auto_mode: bool = False):
        self.auto_mode = auto_mode
        self.project_root = PROJECT_ROOT
        self.dataset_dir = self.project_root / "dataset"
        self.models_dir = self.project_root / "models"
        self.config: Dict[str, Any] = {}

    def print_header(self, text: str):
        """Print a formatted header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70)

    def print_step(self, step: int, total: int, text: str):
        """Print a step indicator"""
        print(f"\n[Step {step}/{total}] {text}")
        print("-" * 70)

    def ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question"""
        if self.auto_mode:
            return default

        default_str = "Y/n" if default else "y/N"
        while True:
            response = input(f"{question} [{default_str}]: ").strip().lower()
            if not response:
                return default
            if response in ['y', 'yes']:
                return True
            if response in ['n', 'no']:
                return False
            print("Please enter 'y' or 'n'")

    def ask_integer(self, question: str, default: int, min_val: int = 1, max_val: int = 10000) -> int:
        """Ask for an integer value"""
        if self.auto_mode:
            return default

        while True:
            response = input(f"{question} [default: {default}]: ").strip()
            if not response:
                return default
            try:
                value = int(response)
                if min_val <= value <= max_val:
                    return value
                print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid integer")

    def ask_choice(self, question: str, choices: list, default: int = 0) -> Any:
        """Ask user to choose from a list"""
        if self.auto_mode:
            return choices[default]

        print(question)
        for i, choice in enumerate(choices):
            marker = "→" if i == default else " "
            print(f"  {marker} {i+1}. {choice}")

        while True:
            response = input(f"Choose [1-{len(choices)}] [default: {default+1}]: ").strip()
            if not response:
                return choices[default]
            try:
                idx = int(response) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
                print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print("Please enter a valid number")

    def validate_dataset(self) -> bool:
        """Validate that the dataset exists and is properly formatted"""
        self.print_step(1, 7, "Validating Dataset")

        # Check if dataset directory exists
        if not self.dataset_dir.exists():
            print(f"❌ Dataset directory not found: {self.dataset_dir}")
            print("\nPlease create the dataset first using:")
            print("  python tools/dataset_creator.py")
            return False

        print(f"✓ Found dataset directory: {self.dataset_dir}")

        # Check for data.yaml
        yaml_path = self.dataset_dir / "data.yaml"
        if not yaml_path.exists():
            print(f"❌ Dataset configuration not found: {yaml_path}")
            print("\nPlease create data.yaml with your class definitions")
            return False

        print(f"✓ Found dataset configuration: {yaml_path}")

        # Read data.yaml to get actual paths
        import yaml as yaml_lib
        with open(yaml_path) as f:
            data_config = yaml_lib.safe_load(f)

        # Check for training images (support both structures)
        # Try: train/images/ (Roboflow format)
        train_images = self.dataset_dir / "train" / "images"
        if not train_images.exists():
            # Try: images/train/ (alternative format)
            train_images = self.dataset_dir / "images" / "train"

        if not train_images.exists() or not list(train_images.glob("*")):
            print(f"❌ No training images found in: {train_images}")
            return False

        train_count = len(list(train_images.glob("*.jpg")) + list(train_images.glob("*.png")))
        print(f"✓ Found {train_count} training images")

        # Check for training labels
        train_labels = self.dataset_dir / "train" / "labels"
        if not train_labels.exists():
            train_labels = self.dataset_dir / "labels" / "train"

        if not train_labels.exists() or not list(train_labels.glob("*.txt")):
            print(f"❌ No training labels found in: {train_labels}")
            print("\nPlease annotate your dataset using Roboflow or LabelImg")
            return False

        label_count = len(list(train_labels.glob("*.txt")))
        print(f"✓ Found {label_count} training labels")

        # Warn if counts don't match
        if train_count != label_count:
            print(f"⚠️  WARNING: Image count ({train_count}) doesn't match label count ({label_count})")
            if not self.ask_yes_no("Continue anyway?", default=False):
                return False

        # Check for validation set
        val_images = self.dataset_dir / "valid" / "images"
        if not val_images.exists():
            val_images = self.dataset_dir / "images" / "val"

        if val_images.exists() and list(val_images.glob("*")):
            val_count = len(list(val_images.glob("*.jpg")) + list(val_images.glob("*.png")))
            print(f"✓ Found {val_count} validation images")
        else:
            print("⚠️  No validation set found - will use train/val split")

        print("\n✅ Dataset validation passed!")
        return True

    def configure_training(self):
        """Configure training parameters"""
        self.print_step(2, 7, "Configure Training Parameters")

        print("\nTraining configuration options:")
        print("  • Epochs: Number of complete passes through the dataset")
        print("  • Batch size: Number of images processed at once (larger = faster but needs more memory)")
        print("  • Image size: Resolution for training (640 is standard, 1280 for higher accuracy)")
        print("  • Device: cpu (slow but works everywhere) or cuda (fast, needs GPU)")

        # Epochs
        self.config['epochs'] = self.ask_integer(
            "\nHow many epochs to train?",
            default=100,
            min_val=10,
            max_val=1000
        )

        # Batch size
        available_memory = "unknown"
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            available_memory = f"{gpu_mem_gb:.1f}GB GPU"
            recommended_batch = min(32, max(8, int(gpu_mem_gb / 2)))
        elif torch.backends.mps.is_available():
            # Apple Silicon - use unified memory (assume 32GB+)
            available_memory = "Apple Silicon (MPS)"
            recommended_batch = 8  # Smaller batch size to avoid tensor size mismatches
        else:
            recommended_batch = 8

        print(f"\nAvailable memory: {available_memory}")
        print(f"Recommended batch size: {recommended_batch}")

        self.config['batch'] = self.ask_integer(
            "Batch size?",
            default=recommended_batch,
            min_val=1,
            max_val=128
        )

        # Image size
        img_size = self.ask_choice(
            "\nTraining image size?",
            [640, 1280],
            default=0
        )
        self.config['imgsz'] = img_size

        # Device
        if torch.cuda.is_available():
            print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
            device = self.ask_choice(
                "Which device to use?",
                ['cuda (GPU - Recommended)', 'cpu'],
                default=0
            )
            self.config['device'] = device.split()[0]
        elif torch.backends.mps.is_available():
            print(f"\n✓ Apple Silicon GPU detected (MPS)")
            device = self.ask_choice(
                "Which device to use?",
                ['mps (Apple GPU - Recommended)', 'cpu'],
                default=0
            )
            self.config['device'] = device.split()[0]
        else:
            print("\nℹ️  No GPU detected, will use CPU (this will be slower)")
            self.config['device'] = 'cpu'

        # Patience (early stopping)
        self.config['patience'] = 20

        # Project name
        print("\nTraining results will be saved to: models/training_runs/")
        if not self.auto_mode:
            run_name = input("Run name [default: yolov8n_pool_v1]: ").strip()
            self.config['name'] = run_name if run_name else 'yolov8n_pool_v1'
        else:
            self.config['name'] = 'yolov8n_pool_v1'

        print("\n📋 Training Configuration:")
        print(f"  • Epochs: {self.config['epochs']}")
        print(f"  • Batch size: {self.config['batch']}")
        print(f"  • Image size: {self.config['imgsz']}x{self.config['imgsz']}")
        print(f"  • Device: {self.config['device']}")
        print(f"  • Run name: {self.config['name']}")

    def download_base_model(self) -> Path:
        """Download pretrained YOLOv8-nano model"""
        self.print_step(3, 7, "Download Base Model")

        base_model = self.models_dir / "yolov8n.pt"

        if base_model.exists():
            print(f"✓ Base model already downloaded: {base_model}")
            return base_model

        print("Downloading YOLOv8-nano pretrained weights...")
        print("This may take a few minutes on first run...")

        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)

        try:
            # YOLO will auto-download to default location, we'll use that
            print("✓ Base model ready")
            return Path("yolov8n.pt")  # Use YOLO's default path
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            sys.exit(1)

    def train_model(self, base_model: Path):
        """Train the YOLO model"""
        self.print_step(4, 7, "Training Model")

        print("\n🚀 Starting training...")
        print(f"   This will take a while ({self.config['epochs']} epochs)")
        print("   You can monitor progress in real-time below")
        print("   Press Ctrl+C to stop training early (model will be saved)\n")

        # Initialize model
        model = YOLO(str(base_model))

        # Start training
        try:
            results = model.train(
                data=str(self.dataset_dir / "data.yaml"),
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                device=self.config['device'],
                patience=self.config['patience'],
                save=True,
                project=str(self.models_dir / "training_runs"),
                name=self.config['name'],
                verbose=True,
                plots=True,  # Generate training plots
            )

            print("\n✅ Training completed successfully!")

            # Store results path
            self.config['results_dir'] = Path(results.save_dir)
            self.config['best_model'] = self.config['results_dir'] / "weights" / "best.pt"

            return results

        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            print("Partial model has been saved")
            return None
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def analyze_results(self):
        """Analyze training results"""
        self.print_step(5, 7, "Analyzing Results")

        if not hasattr(self, 'config') or 'results_dir' not in self.config:
            print("⚠️  No training results to analyze")
            return

        results_dir = self.config['results_dir']

        print(f"\n📊 Training results saved to: {results_dir}")

        # Check for training plots
        plots = list(results_dir.glob("*.png"))
        if plots:
            print(f"\n✓ Generated {len(plots)} training plots:")
            for plot in sorted(plots):
                print(f"  • {plot.name}")

        # Check for best model
        best_model = results_dir / "weights" / "best.pt"
        if best_model.exists():
            size_mb = best_model.stat().st_size / 1e6
            print(f"\n✓ Best model saved: {best_model}")
            print(f"  Size: {size_mb:.2f} MB")

            # Load and show metrics
            try:
                model = YOLO(str(best_model))
                print("\n📈 Model Metrics:")
                print(f"  • Model loaded successfully")
            except Exception as e:
                print(f"  ⚠️  Could not load model: {e}")

        # Check for results.csv
        csv_file = results_dir / "results.csv"
        if csv_file.exists():
            print(f"\n✓ Training metrics saved: {csv_file}")

            # Read last line for final metrics
            try:
                with open(csv_file) as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        headers = lines[0].strip().split(',')
                        values = lines[-1].strip().split(',')

                        # Find mAP columns
                        for i, header in enumerate(headers):
                            if 'map50' in header.lower() and i < len(values):
                                print(f"\n  Final mAP@0.5: {float(values[i]):.3f}")
                            elif 'map' in header.lower() and '95' in header and i < len(values):
                                print(f"  Final mAP@0.5-0.95: {float(values[i]):.3f}")
            except Exception as e:
                print(f"  ⚠️  Could not parse metrics: {e}")

    def export_model(self):
        """Export trained model to ONNX format"""
        self.print_step(6, 7, "Export Model to ONNX")

        if 'best_model' not in self.config or not Path(self.config['best_model']).exists():
            print("❌ No trained model found to export")
            return

        print("\nExporting model to ONNX format for production deployment...")
        print("ONNX models are optimized for CPU inference and smaller file size")

        if not self.ask_yes_no("\nExport to ONNX?", default=True):
            print("Skipping ONNX export")
            return

        try:
            model = YOLO(str(self.config['best_model']))

            print("\n🔄 Exporting to ONNX (this may take a minute)...")
            onnx_path = model.export(format='onnx', imgsz=self.config['imgsz'])

            print(f"\n✅ ONNX model exported: {onnx_path}")

            # Copy to models directory
            dest_path = self.models_dir / f"yolov8n-pool-{self.config['name']}.onnx"
            shutil.copy(onnx_path, dest_path)

            size_mb = dest_path.stat().st_size / 1e6
            print(f"✓ Copied to: {dest_path}")
            print(f"  Size: {size_mb:.2f} MB")

            if size_mb > 10:
                print(f"  ⚠️  Model is larger than target 10MB")
            else:
                print(f"  ✓ Model size is within target (<10MB)")

            self.config['onnx_model'] = dest_path

        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            import traceback
            traceback.print_exc()

    def show_next_steps(self):
        """Show next steps for using the trained model"""
        self.print_step(7, 7, "Next Steps")

        print("\n🎉 Training pipeline completed!")
        print("\n📝 Next steps:\n")

        if 'onnx_model' in self.config:
            print(f"1. Test your model:")
            print(f"   python tools/test_yolo_model.py {self.config['onnx_model']}")
            print()

        if 'best_model' in self.config:
            print(f"2. Validate model performance:")
            print(f"   python tools/benchmark_detectors.py --yolo-model {self.config['best_model']}")
            print()

        print(f"3. Review training plots in:")
        if 'results_dir' in self.config:
            print(f"   {self.config['results_dir']}")
        print()

        print("4. Integrate model into vision system:")
        print("   • Copy ONNX model to models/yolov8n-pool.onnx")
        print("   • Update config: detection_backend = 'yolo'")
        print("   • Restart backend")
        print()

        print("5. If results aren't good enough:")
        print("   • Collect more training data")
        print("   • Improve annotation quality")
        print("   • Try training for more epochs")
        print("   • Experiment with different image sizes")
        print()

        print("📚 Documentation:")
        print("   • Training guide: docs/YOLO_TRAINING_GUIDE.md")
        print("   • Model documentation: models/README.md")
        print()

    def save_config(self):
        """Save training configuration"""
        if not hasattr(self, 'config') or not self.config:
            return

        config_file = self.models_dir / "training_runs" / self.config['name'] / "training_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert Path objects to strings for JSON serialization
        json_config = {}
        for key, value in self.config.items():
            if isinstance(value, Path):
                json_config[key] = str(value)
            else:
                json_config[key] = value

        with open(config_file, 'w') as f:
            json.dump(json_config, f, indent=2)

        print(f"\n✓ Configuration saved: {config_file}")

    def run(self):
        """Run the complete training pipeline"""
        self.print_header("🎱 YOLO Training Pipeline for Billiards Detection")

        print("\nThis wizard will guide you through training a YOLOv8 model to detect:")
        print("  • Billiard balls (cue ball, solids, stripes, 8-ball)")
        print("  • Cue stick")
        print("  • Optional: Table corners and pockets")

        if self.auto_mode:
            print("\n🤖 Running in AUTO mode (using default values)")
        else:
            print("\nYou'll be asked to configure training parameters at each step.")
            print("Press Ctrl+C at any time to cancel.")

        try:
            # Step 1: Validate dataset
            if not self.validate_dataset():
                return 1

            # Step 2: Configure training
            self.configure_training()

            # Confirm before starting
            if not self.auto_mode:
                print("\n" + "="*70)
                if not self.ask_yes_no("\nReady to start training?", default=True):
                    print("Training cancelled")
                    return 0

            # Step 3: Download base model
            base_model = self.download_base_model()

            # Step 4: Train model
            results = self.train_model(base_model)

            if results is None:
                return 1

            # Step 5: Analyze results
            self.analyze_results()

            # Step 6: Export to ONNX
            self.export_model()

            # Step 7: Show next steps
            self.show_next_steps()

            # Save configuration
            self.save_config()

            return 0

        except KeyboardInterrupt:
            print("\n\n⚠️  Training cancelled by user")
            return 1
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Interactive YOLO training for billiards detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first-time users)
  python tools/train_yolo.py

  # Auto mode (uses defaults, good for scripts)
  python tools/train_yolo.py --auto

  # View this help
  python tools/train_yolo.py --help
        """
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help='Run in automatic mode with default settings'
    )

    args = parser.parse_args()

    trainer = YOLOTrainer(auto_mode=args.auto)
    return trainer.run()


if __name__ == "__main__":
    sys.exit(main())
