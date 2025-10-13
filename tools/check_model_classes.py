#!/usr/bin/env python3
"""Check classes in all training runs to find the billiards model."""

from pathlib import Path
from ultralytics import YOLO

training_runs = Path("models/training_runs")

for run_dir in sorted(training_runs.glob("yolov8n_pool_v*")):
    best_pt = run_dir / "weights" / "best.pt"

    if not best_pt.exists():
        continue

    print(f"\n{'='*60}")
    print(f"Run: {run_dir.name}")
    print(f"{'='*60}")

    try:
        model = YOLO(str(best_pt))

        # Show all classes
        print(f"Total classes: {len(model.names)}")
        print("Classes:")
        for cls_id, cls_name in model.names.items():
            print(f"  {cls_id}: {cls_name}")

        # Check if it's a billiards model
        class_names = set(model.names.values())
        ball_keywords = {'ball', 'cue', 'eight', 'solid', 'stripe'}
        is_billiards = bool(ball_keywords & {c.lower() for c in class_names})

        if is_billiards:
            print("\n✓ THIS LOOKS LIKE A BILLIARDS MODEL!")
        else:
            print("\n✗ Not a billiards model")

    except Exception as e:
        print(f"Error loading model: {e}")
