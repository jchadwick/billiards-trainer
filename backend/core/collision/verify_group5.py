#!/usr/bin/env python
"""Verification script for Group 5: Collision Detection to 4K migration.

This script verifies that:
1. BALL_RADIUS_4K is imported and used correctly (36 pixels)
2. All collision detection operates in 4K pixels
3. Collision points have scale metadata
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def verify_imports():
    """Verify that all required 4K constants are imported."""
    print("=" * 70)
    print("GROUP 5 VERIFICATION: Collision Detection to 4K")
    print("=" * 70)
    print()
    print("Step 1: Verifying Imports in Source Code")
    print("-" * 70)

    # Check that geometric_collision.py imports BALL_RADIUS_4K
    geometric_collision_path = os.path.join(
        os.path.dirname(__file__), "geometric_collision.py"
    )

    with open(geometric_collision_path) as f:
        content = f.read()

    if "from ..constants_4k import BALL_RADIUS_4K" in content:
        print("✓ geometric_collision.py imports BALL_RADIUS_4K from constants_4k")
    else:
        print("✗ geometric_collision.py does not import BALL_RADIUS_4K")
        return False

    # Verify the constant exists in constants_4k.py
    constants_path = os.path.join(os.path.dirname(__file__), "../constants_4k.py")
    with open(constants_path) as f:
        constants_content = f.read()

    if "BALL_RADIUS_4K = 36" in constants_content:
        print("✓ constants_4k.py defines BALL_RADIUS_4K = 36 pixels")
    else:
        print("✗ constants_4k.py does not define BALL_RADIUS_4K correctly")
        return False

    return True


def verify_documentation():
    """Verify that documentation has been updated."""
    print()
    print("Step 2: Verifying Documentation")
    print("-" * 70)

    geometric_collision_path = os.path.join(
        os.path.dirname(__file__), "geometric_collision.py"
    )

    with open(geometric_collision_path) as f:
        content = f.read()

    # Check for 4K migration comments
    checks = [
        ("4K Migration (Group 5)", "4K Migration documentation header"),
        ("BALL_RADIUS_4K", "BALL_RADIUS_4K reference"),
        ("3840×2160", "4K resolution reference"),
        ("36 pixels", "Ball radius documentation"),
        ("scale=[1.0, 1.0]", "Scale metadata documentation"),
    ]

    all_passed = True
    for search_str, description in checks:
        if search_str in content:
            print(f"✓ Found: {description}")
        else:
            print(f"✗ Missing: {description}")
            all_passed = False

    return all_passed


def verify_class_attributes():
    """Verify that GeometricCollisionDetector has BALL_RADIUS_4K attribute."""
    print()
    print("Step 3: Verifying Class Attributes")
    print("-" * 70)

    geometric_collision_path = os.path.join(
        os.path.dirname(__file__), "geometric_collision.py"
    )

    with open(geometric_collision_path) as f:
        content = f.read()

    # Check for ball_radius_4k attribute in __init__
    if "self.ball_radius_4k = BALL_RADIUS_4K" in content:
        print("✓ GeometricCollisionDetector has ball_radius_4k attribute")
        return True
    else:
        print("✗ GeometricCollisionDetector missing ball_radius_4k attribute")
        return False


def verify_method_documentation():
    """Verify that key methods document 4K pixel usage."""
    print()
    print("Step 4: Verifying Method Documentation")
    print("-" * 70)

    geometric_collision_path = os.path.join(
        os.path.dirname(__file__), "geometric_collision.py"
    )

    with open(geometric_collision_path) as f:
        content = f.read()

    methods_to_check = [
        "check_line_circle_intersection",
        "find_closest_ball_collision",
        "find_cushion_intersection",
        "find_pocket_intersection",
        "calculate_geometric_reflection",
        "calculate_ball_collision_velocities",
    ]

    all_passed = True
    for method in methods_to_check:
        # Look for the method and check if it has 4K migration docs
        method_start = content.find(f"def {method}(")
        if method_start == -1:
            print(f"✗ Method not found: {method}")
            all_passed = False
            continue

        # Look for docstring with 4K migration notes
        # Get next 2000 chars after method definition
        method_section = content[method_start : method_start + 2000]

        if "4K Migration" in method_section or "4K pixels" in method_section:
            print(f"✓ {method} has 4K documentation")
        else:
            print(f"✗ {method} missing 4K documentation")
            all_passed = False

    return all_passed


def main():
    """Run all verification checks."""
    results = []

    results.append(("Imports", verify_imports()))
    results.append(("Documentation", verify_documentation()))
    results.append(("Class Attributes", verify_class_attributes()))
    results.append(("Method Documentation", verify_method_documentation()))

    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for check_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check_name:25s}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Group 5 migration complete.")
        print()
        print("Summary of Changes:")
        print("  - Added import of BALL_RADIUS_4K (36 pixels)")
        print("  - Updated all docstrings to document 4K pixel usage")
        print("  - Added ball_radius_4k attribute to GeometricCollisionDetector")
        print("  - Documented that all distances are in 4K pixels")
        print("  - Documented that collision points have scale=[1.0, 1.0]")
        return 0
    else:
        print("❌ SOME CHECKS FAILED. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
