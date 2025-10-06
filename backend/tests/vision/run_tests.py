#!/usr/bin/env python
"""Simple test runner for detector_adapter tests without pytest complications."""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

# Get all test functions
import inspect

# Now we can import the test module
import test_detector_adapter

test_functions = [
    (name, func)
    for name, func in inspect.getmembers(test_detector_adapter)
    if name.startswith("test_") and callable(func)
]

# Run each test
passed = 0
failed = 0
errors = []

print(f"Running {len(test_functions)} tests from test_detector_adapter.py\n")
print("=" * 70)

for name, func in test_functions:
    try:
        # Get function signature to determine if it needs fixtures
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Prepare arguments if fixtures are needed
        kwargs = {}
        if "image_shape" in params:
            kwargs["image_shape"] = (1080, 1920)
        if "class_names" in params:
            kwargs["class_names"] = [
                "cue",
                "solid_1",
                "solid_2",
                "solid_3",
                "solid_4",
                "solid_5",
                "solid_6",
                "solid_7",
                "eight",
                "stripe_9",
                "stripe_10",
                "stripe_11",
                "stripe_12",
                "stripe_13",
                "stripe_14",
                "stripe_15",
                "cue_stick",
            ]
        if "sample_detection_cue" in params:
            kwargs["sample_detection_cue"] = {
                "bbox": [930, 510, 990, 570],
                "confidence": 0.95,
                "class_id": 0,
                "class_name": "cue",
            }
        if "sample_detection_solid" in params:
            kwargs["sample_detection_solid"] = {
                "bbox": [800, 400, 860, 460],
                "confidence": 0.88,
                "class_id": 3,
                "class_name": "solid_3",
            }
        if "sample_detection_stripe" in params:
            kwargs["sample_detection_stripe"] = {
                "bbox": [1100, 600, 1160, 660],
                "confidence": 0.82,
                "class_id": 12,
                "class_name": "stripe_12",
            }
        if "sample_detection_eight" in params:
            kwargs["sample_detection_eight"] = {
                "bbox": [1000, 700, 1060, 760],
                "confidence": 0.91,
                "class_id": 8,
                "class_name": "eight",
            }
        if "sample_ball_bbox_xywh" in params:
            kwargs["sample_ball_bbox_xywh"] = [930, 510, 60, 60]
        if "sample_ball_bbox_normalized" in params:
            kwargs["sample_ball_bbox_normalized"] = [0.484375, 0.4722, 0.03125, 0.0556]
        if "sample_cue_bbox" in params:
            kwargs["sample_cue_bbox"] = [700, 390, 200, 20]

        func(**kwargs)
        print(f"✓ {name}")
        passed += 1
    except Exception as e:
        print(f"✗ {name}")
        print(f"  Error: {type(e).__name__}: {e}")
        failed += 1
        errors.append((name, e))

print("=" * 70)
print(f"\nResults: {passed} passed, {failed} failed out of {len(test_functions)} tests")

if errors:
    print("\nFailed tests:")
    for name, error in errors:
        print(f"  - {name}: {error}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
