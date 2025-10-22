#!/usr/bin/env python3
"""Verification script for Group 1 (Foundation & Constants) implementation.

This script demonstrates that all Group 1 deliverables are working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def verify_constants():
    """Verify constants_4k module."""
    print("=" * 70)
    print("VERIFYING CONSTANTS_4K MODULE")
    print("=" * 70)

    from backend.core.constants_4k import (
        BALL_DIAMETER_4K,
        BALL_RADIUS_4K,
        CANONICAL_HEIGHT,
        CANONICAL_RESOLUTION,
        CANONICAL_WIDTH,
        CUSHION_WIDTH_4K,
        POCKET_POSITIONS_4K,
        POCKET_RADIUS_4K,
        TABLE_BOTTOM_4K,
        TABLE_CENTER_4K,
        TABLE_HEIGHT_4K,
        TABLE_LEFT_4K,
        TABLE_RIGHT_4K,
        TABLE_TOP_4K,
        TABLE_WIDTH_4K,
        get_table_bounds_4k,
        is_on_table,
        is_valid_4k_coordinate,
    )

    print("\n1. Canonical Resolution:")
    print(f"   CANONICAL_RESOLUTION: {CANONICAL_RESOLUTION}")
    print(f"   CANONICAL_WIDTH: {CANONICAL_WIDTH}")
    print(f"   CANONICAL_HEIGHT: {CANONICAL_HEIGHT}")
    print(
        f"   Aspect Ratio: {CANONICAL_WIDTH / CANONICAL_HEIGHT:.4f} (16:9 = {16/9:.4f})"
    )
    assert CANONICAL_RESOLUTION == (3840, 2160)
    assert abs((CANONICAL_WIDTH / CANONICAL_HEIGHT) - (16 / 9)) < 0.0001
    print("   ✅ Canonical resolution verified")

    print("\n2. Table Dimensions:")
    print(f"   TABLE_WIDTH_4K: {TABLE_WIDTH_4K}")
    print(f"   TABLE_HEIGHT_4K: {TABLE_HEIGHT_4K}")
    print(f"   TABLE_CENTER_4K: {TABLE_CENTER_4K}")
    print(f"   Aspect Ratio: {TABLE_WIDTH_4K / TABLE_HEIGHT_4K:.1f} (expected 2:1)")
    assert TABLE_WIDTH_4K / TABLE_HEIGHT_4K == 2.0
    assert TABLE_CENTER_4K == (1920, 1080)
    print("   ✅ Table dimensions verified")

    print("\n3. Table Bounds:")
    print(f"   LEFT: {TABLE_LEFT_4K}, TOP: {TABLE_TOP_4K}")
    print(f"   RIGHT: {TABLE_RIGHT_4K}, BOTTOM: {TABLE_BOTTOM_4K}")
    print(
        f"   Left Margin: {TABLE_LEFT_4K}px, Right Margin: {CANONICAL_WIDTH - TABLE_RIGHT_4K}px"
    )
    print(
        f"   Top Margin: {TABLE_TOP_4K}px, Bottom Margin: {CANONICAL_HEIGHT - TABLE_BOTTOM_4K}px"
    )
    assert TABLE_LEFT_4K == 320
    assert TABLE_RIGHT_4K == 3520
    assert TABLE_TOP_4K == 280
    assert TABLE_BOTTOM_4K == 1880
    print("   ✅ Table bounds verified")

    print("\n4. Ball Dimensions:")
    print(f"   BALL_RADIUS_4K: {BALL_RADIUS_4K}px")
    print(f"   BALL_DIAMETER_4K: {BALL_DIAMETER_4K}px")
    assert BALL_RADIUS_4K * 2 == BALL_DIAMETER_4K
    print("   ✅ Ball dimensions verified")

    print("\n5. Pocket Dimensions:")
    print(f"   POCKET_RADIUS_4K: {POCKET_RADIUS_4K}px")
    print(f"   Number of pockets: {len(POCKET_POSITIONS_4K)}")
    print("   Pocket positions:")
    for i, pos in enumerate(POCKET_POSITIONS_4K):
        print(f"     Pocket {i}: {pos}")
    assert len(POCKET_POSITIONS_4K) == 6
    print("   ✅ Pocket dimensions verified")

    print("\n6. Validation Helpers:")
    assert is_valid_4k_coordinate(1920, 1080)
    assert not is_valid_4k_coordinate(-1, 1080)
    assert is_on_table(1920, 1080)
    bounds = get_table_bounds_4k()
    print(f"   Table bounds: {bounds}")
    assert bounds == (320, 280, 3520, 1880)
    print("   ✅ Validation helpers verified")

    print("\n" + "=" * 70)
    print("✅ CONSTANTS_4K MODULE VERIFIED SUCCESSFULLY")
    print("=" * 70)


def verify_resolution_converter():
    """Verify resolution_converter module."""
    print("\n" + "=" * 70)
    print("VERIFYING RESOLUTION_CONVERTER MODULE")
    print("=" * 70)

    from backend.core.resolution_converter import ResolutionConverter, from_4k, to_4k

    print("\n1. Scale Calculations:")
    scale_1080p = ResolutionConverter.calculate_scale_to_4k((1920, 1080))
    scale_720p = ResolutionConverter.calculate_scale_to_4k((1280, 720))
    print(f"   1080p → 4K scale: {scale_1080p}")
    print(f"   720p → 4K scale: {scale_720p}")
    assert scale_1080p == (2.0, 2.0)
    assert scale_720p == (3.0, 3.0)
    print("   ✅ Scale calculations verified")

    print("\n2. Coordinate Conversion to 4K:")
    x_4k, y_4k = ResolutionConverter.scale_to_4k(960, 540, (1920, 1080))
    print(f"   1080p (960, 540) → 4K ({x_4k}, {y_4k})")
    assert x_4k == 1920.0
    assert y_4k == 1080.0

    x_4k, y_4k = ResolutionConverter.scale_to_4k(640, 360, (1280, 720))
    print(f"   720p (640, 360) → 4K ({x_4k}, {y_4k})")
    assert x_4k == 1920.0
    assert y_4k == 1080.0
    print("   ✅ Coordinate conversion to 4K verified")

    print("\n3. Coordinate Conversion from 4K:")
    x, y = ResolutionConverter.scale_from_4k(1920, 1080, (1920, 1080))
    print(f"   4K (1920, 1080) → 1080p ({x}, {y})")
    assert x == 960.0
    assert y == 540.0
    print("   ✅ Coordinate conversion from 4K verified")

    print("\n4. Distance Scaling:")
    distance_4k = ResolutionConverter.scale_distance_to_4k(18, (1920, 1080))
    print(f"   1080p 18px → 4K {distance_4k}px")
    assert distance_4k == 36.0

    distance = ResolutionConverter.scale_distance_from_4k(36, (1920, 1080))
    print(f"   4K 36px → 1080p {distance}px")
    assert distance == 18.0
    print("   ✅ Distance scaling verified")

    print("\n5. Round-Trip Conversion:")
    original_x, original_y = 960.5, 540.25
    x_4k, y_4k = ResolutionConverter.scale_to_4k(original_x, original_y, (1920, 1080))
    x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, (1920, 1080))
    error_x = abs(x - original_x)
    error_y = abs(y - original_y)
    print(f"   Original: ({original_x}, {original_y})")
    print(f"   Round-trip: ({x}, {y})")
    print(f"   Error: ({error_x:.10f}, {error_y:.10f})")
    assert error_x < 1e-6
    assert error_y < 1e-6
    print("   ✅ Round-trip conversion verified (< 1e-6 error)")

    print("\n6. Convenience Functions:")
    x_4k, y_4k = to_4k(960, 540, (1920, 1080))
    assert x_4k == 1920.0
    assert y_4k == 1080.0

    x, y = from_4k(1920, 1080, (1920, 1080))
    assert x == 960.0
    assert y == 540.0
    print("   ✅ Convenience functions verified")

    print("\n7. Helper Methods:")
    assert ResolutionConverter.is_4k_canonical((3840, 2160))
    assert not ResolutionConverter.is_4k_canonical((1920, 1080))
    ratio = ResolutionConverter.get_aspect_ratio((1920, 1080))
    print(f"   1080p aspect ratio: {ratio:.4f} (16:9 = {16/9:.4f})")
    assert abs(ratio - 16 / 9) < 0.0001
    print("   ✅ Helper methods verified")

    print("\n" + "=" * 70)
    print("✅ RESOLUTION_CONVERTER MODULE VERIFIED SUCCESSFULLY")
    print("=" * 70)


def verify_imports():
    """Verify imports from backend.core."""
    print("\n" + "=" * 70)
    print("VERIFYING IMPORTS FROM backend.core")
    print("=" * 70)

    from backend.core import (
        BALL_RADIUS_4K,
        CANONICAL_RESOLUTION,
        POCKET_RADIUS_4K,
        TABLE_CENTER_4K,
        TABLE_HEIGHT_4K,
        TABLE_WIDTH_4K,
        ResolutionConverter,
    )

    print("\n1. Constants:")
    print(f"   CANONICAL_RESOLUTION: {CANONICAL_RESOLUTION}")
    print(f"   TABLE_WIDTH_4K: {TABLE_WIDTH_4K}")
    print(f"   TABLE_HEIGHT_4K: {TABLE_HEIGHT_4K}")
    print(f"   TABLE_CENTER_4K: {TABLE_CENTER_4K}")
    print(f"   BALL_RADIUS_4K: {BALL_RADIUS_4K}")
    print(f"   POCKET_RADIUS_4K: {POCKET_RADIUS_4K}")
    print("   ✅ Constants imported successfully")

    print("\n2. ResolutionConverter:")
    scale = ResolutionConverter.calculate_scale_to_4k((1920, 1080))
    print(f"   ResolutionConverter.calculate_scale_to_4k((1920, 1080)): {scale}")
    assert scale == (2.0, 2.0)
    print("   ✅ ResolutionConverter imported successfully")

    print("\n" + "=" * 70)
    print("✅ ALL IMPORTS VERIFIED SUCCESSFULLY")
    print("=" * 70)


def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + " " * 15 + "4K MIGRATION - GROUP 1 VERIFICATION" + " " * 18 + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        verify_constants()
        verify_resolution_converter()
        verify_imports()

        print("\n\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print(
            "║" + " " * 10 + "🎉 ALL GROUP 1 VERIFICATIONS PASSED! 🎉" + " " * 17 + "║"
        )
        print("║" + " " * 68 + "║")
        print(
            "║" + " " * 15 + "Group 2 (Vector2D) is ready to proceed" + " " * 14 + "║"
        )
        print("║" + " " * 68 + "║")
        print("╚" + "=" * 68 + "╝")
        print("\n")

        return 0

    except Exception as e:
        print("\n\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + " " * 20 + "❌ VERIFICATION FAILED ❌" + " " * 23 + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "=" * 68 + "╝")
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
