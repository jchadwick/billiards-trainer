#!/usr/bin/env python3
"""
Fix corrupted YOLO dataset labels

Finds and fixes common label file issues:
- Boxes outside image bounds (x, y, w, h not in [0, 1])
- Invalid class IDs
- Malformed lines
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
dataset_dir = PROJECT_ROOT / "dataset"

def validate_and_fix_label(label_path: Path, num_classes: int = 16):
    """Validate and fix a single label file"""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        valid_lines = []
        fixed_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split()
                if len(parts) != 5:
                    print(f"  ⚠️  Skipping malformed line: {line}")
                    fixed_count += 1
                    continue

                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                # Validate class ID
                if class_id < 0 or class_id >= num_classes:
                    print(f"  ⚠️  Invalid class {class_id}, skipping")
                    fixed_count += 1
                    continue

                # Validate and clamp coordinates
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                # Check if box is valid (within bounds)
                if (x - w/2 < 0 or x + w/2 > 1 or
                    y - h/2 < 0 or y + h/2 > 1):
                    # Clamp box to stay within image
                    w = min(w, 2*x, 2*(1-x))
                    h = min(h, 2*y, 2*(1-y))
                    fixed_count += 1

                # Skip tiny boxes (likely errors)
                if w < 0.01 or h < 0.01:
                    print(f"  ⚠️  Skipping tiny box: w={w}, h={h}")
                    fixed_count += 1
                    continue

                valid_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            except (ValueError, IndexError) as e:
                print(f"  ⚠️  Error parsing line '{line}': {e}")
                fixed_count += 1
                continue

        # Write back if we have valid lines
        if valid_lines:
            with open(label_path, 'w') as f:
                f.writelines(valid_lines)
            return fixed_count, len(valid_lines)
        else:
            # Delete empty label file
            label_path.unlink()
            print(f"  ❌ Deleted empty label file")
            return fixed_count, 0

    except Exception as e:
        print(f"  ❌ Error processing {label_path}: {e}")
        return 0, 0


def main():
    print("="*70)
    print("  🔧 YOLO Dataset Label Fixer")
    print("="*70)

    # Find all label files
    label_dirs = [
        dataset_dir / "train" / "labels",
        dataset_dir / "valid" / "labels",
        dataset_dir / "test" / "labels",
    ]

    total_files = 0
    total_fixed = 0
    total_boxes = 0

    for label_dir in label_dirs:
        if not label_dir.exists():
            continue

        print(f"\n📂 Processing {label_dir}")

        label_files = list(label_dir.glob("*.txt"))
        print(f"   Found {len(label_files)} label files")

        for label_file in label_files:
            fixed, boxes = validate_and_fix_label(label_file)
            if fixed > 0:
                print(f"   ✓ {label_file.name}: fixed {fixed} issues, {boxes} valid boxes")
            total_files += 1
            total_fixed += fixed
            total_boxes += boxes

    print("\n" + "="*70)
    print(f"✅ Done!")
    print(f"   Processed: {total_files} files")
    print(f"   Fixed: {total_fixed} issues")
    print(f"   Valid boxes: {total_boxes}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
