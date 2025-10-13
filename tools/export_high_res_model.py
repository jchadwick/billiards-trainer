#!/usr/bin/env python3
"""Export YOLO model with higher resolution for better detection quality.

This script exports your trained YOLO model to ONNX format with a larger
input size (1280x1280) to preserve detail from your 1280x720 camera.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ultralytics import YOLO


def export_high_res_model(
    model_path: str,
    output_dir: str = "models",
    imgsz: int = 1280,
    format: str = "onnx"
):
    """Export YOLO model with specified input size.

    Args:
        model_path: Path to .pt model file
        output_dir: Directory to save exported model
        imgsz: Input image size (default: 1280)
        format: Export format (default: onnx)
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"\nExporting to {format.upper()} with input size {imgsz}x{imgsz}...")
    print("This may take a minute...")

    # Export model
    export_path = model.export(
        format=format,
        imgsz=imgsz,
        optimize=True,  # Optimize for inference
        simplify=True,  # Simplify ONNX model
    )

    print(f"\n✓ Model exported successfully!")
    print(f"  Exported to: {export_path}")
    print(f"  Input size: {imgsz}x{imgsz}")

    # Get file size
    size_mb = Path(export_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    print(f"\nNext steps:")
    print(f"1. Update your config to use this model:")
    print(f'   "yolo_model_path": "{export_path}"')
    print(f"\n2. Or copy it to replace your current model:")
    print(f"   cp {export_path} models/yolov8n-pool.onnx")

    return export_path


if __name__ == "__main__":
    # Find the best model from latest training run
    import glob

    training_runs = Path("models/training_runs")

    if not training_runs.exists():
        print("Error: No training runs found in models/training_runs/")
        sys.exit(1)

    # Get all training runs sorted by version number
    all_runs = sorted(training_runs.glob("yolov8n_pool_v*"))

    if not all_runs:
        print("Error: No YOLO training runs found")
        sys.exit(1)

    print("Available training runs:")
    for i, run in enumerate(all_runs, 1):
        best_pt = run / "weights" / "best.pt"
        if best_pt.exists():
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            print(f"  {i}. {run.name} ({size_mb:.1f} MB)")

    # Find the latest run with a valid best.pt
    latest_run = None
    best_model = None

    for run in reversed(all_runs):  # Start from newest
        candidate = run / "weights" / "best.pt"
        if candidate.exists():
            latest_run = run
            best_model = candidate
            break

    if not best_model:
        print(f"\nError: No best.pt found in any training run")
        sys.exit(1)

    print(f"\n→ Using latest model: {latest_run.name}")

    # Export with 1280x1280 input size (matches camera width, square for YOLO)
    export_high_res_model(
        model_path=str(best_model),
        imgsz=1280,
        format="onnx"
    )
