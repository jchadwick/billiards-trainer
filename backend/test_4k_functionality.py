#!/usr/bin/env python3
"""Comprehensive test script for 4K coordinate system migration.

This script verifies:
1. Vector2D creation and conversion
2. BallState 4K coordinate handling
3. Resolution-based scaling
4. Round-trip conversions
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

print("=" * 80)
print("4K COORDINATE SYSTEM MIGRATION TEST")
print("=" * 80)
print()

# Test 1: Vector2D basics
print("TEST 1: Vector2D Factory Methods")
print("-" * 80)
try:
    from core.constants_4k import CANONICAL_HEIGHT, CANONICAL_WIDTH
    from core.coordinates import Vector2D

    # Create vector in 4K canonical space
    v1 = Vector2D.from_4k(1920, 1080)
    print(f"✓ Created 4K vector: {v1}")
    print(f"  - Position: ({v1.x}, {v1.y})")
    print(f"  - Scale: {v1.scale}")
    assert v1.scale == (1.0, 1.0), f"Expected scale (1.0, 1.0), got {v1.scale}"
    print("  ✓ Scale is canonical (1.0, 1.0)")

    # Create vector from 1080p resolution
    v2 = Vector2D.from_resolution(960, 540, (1920, 1080))
    print(f"\n✓ Created 1080p vector: {v2}")
    print(f"  - Position: ({v2.x}, {v2.y})")
    print(f"  - Scale: {v2.scale}")
    assert v2.scale == (2.0, 2.0), f"Expected scale (2.0, 2.0), got {v2.scale}"
    print("  ✓ Scale is correct (2.0, 2.0)")

    # Convert to 4K canonical
    v2_4k = v2.to_4k_canonical()
    print(f"\n✓ Converted to 4K: {v2_4k}")
    print(f"  - Position: ({v2_4k.x}, {v2_4k.y})")
    print(f"  - Scale: {v2_4k.scale}")
    assert abs(v2_4k.x - 1920) < 0.01, f"Expected x=1920, got {v2_4k.x}"
    assert abs(v2_4k.y - 1080) < 0.01, f"Expected y=1080, got {v2_4k.y}"
    assert v2_4k.scale == (1.0, 1.0), f"Expected canonical scale, got {v2_4k.scale}"
    print("  ✓ Conversion correct!")

    print("\n✅ TEST 1 PASSED: Vector2D factory methods work correctly\n")
except Exception as e:
    print(f"\n❌ TEST 1 FAILED: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 2: BallState 4K handling
print("TEST 2: BallState 4K Coordinate Handling")
print("-" * 80)
try:
    from core.models import BallState

    # Create ball in 4K canonical space
    ball1 = BallState.from_4k("ball_1", x=1920, y=1080, number=1)
    print(f"✓ Created 4K ball: {ball1.id}")
    print(f"  - Position: ({ball1.position.x}, {ball1.position.y})")
    print(f"  - Scale: {ball1.position.scale}")
    assert ball1.position.scale == (
        1.0,
        1.0,
    ), f"Expected canonical scale, got {ball1.position.scale}"
    print("  ✓ Position has canonical scale")

    # Create ball from 720p resolution
    ball2 = BallState.from_resolution(
        "ball_2", x=640, y=360, resolution=(1280, 720), number=2
    )
    print(f"\n✓ Created 720p ball: {ball2.id}")
    print(f"  - Position: ({ball2.position.x}, {ball2.position.y})")
    print(f"  - Scale: {ball2.position.scale}")
    assert ball2.position.scale == (
        3.0,
        3.0,
    ), f"Expected scale (3.0, 3.0), got {ball2.position.scale}"
    print("  ✓ Scale is correct (3.0, 3.0)")

    # Convert to 4K
    pos_4k = ball2.position.to_4k_canonical()
    print(f"\n✓ Converted position to 4K: ({pos_4k.x}, {pos_4k.y})")
    assert abs(pos_4k.x - 1920) < 0.01, f"Expected x=1920, got {pos_4k.x}"
    assert abs(pos_4k.y - 1080) < 0.01, f"Expected y=1080, got {pos_4k.y}"
    print("  ✓ Conversion correct!")

    print("\n✅ TEST 2 PASSED: BallState 4K handling works correctly\n")
except Exception as e:
    print(f"\n❌ TEST 2 FAILED: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Round-trip conversions
print("TEST 3: Round-Trip Conversions")
print("-" * 80)
try:
    # Test 1080p -> 4K -> 1080p
    original = Vector2D.from_resolution(500, 300, (1920, 1080))
    print(f"Original 1080p vector: ({original.x}, {original.y}) scale={original.scale}")

    canonical = original.to_4k_canonical()
    print(f"Converted to 4K: ({canonical.x}, {canonical.y}) scale={canonical.scale}")

    back = canonical.to_resolution((1920, 1080))
    print(f"Converted back to 1080p: ({back.x}, {back.y}) scale={back.scale}")

    assert (
        abs(back.x - original.x) < 0.01
    ), f"Round trip x mismatch: {back.x} != {original.x}"
    assert (
        abs(back.y - original.y) < 0.01
    ), f"Round trip y mismatch: {back.y} != {original.y}"
    print("✓ Round-trip 1080p -> 4K -> 1080p successful")

    # Test 720p -> 4K -> 720p
    original_720 = Vector2D.from_resolution(400, 240, (1280, 720))
    print(
        f"\nOriginal 720p vector: ({original_720.x}, {original_720.y}) scale={original_720.scale}"
    )

    canonical_720 = original_720.to_4k_canonical()
    print(
        f"Converted to 4K: ({canonical_720.x}, {canonical_720.y}) scale={canonical_720.scale}"
    )

    back_720 = canonical_720.to_resolution((1280, 720))
    print(
        f"Converted back to 720p: ({back_720.x}, {back_720.y}) scale={back_720.scale}"
    )

    assert (
        abs(back_720.x - original_720.x) < 0.01
    ), f"Round trip x mismatch: {back_720.x} != {original_720.x}"
    assert (
        abs(back_720.y - original_720.y) < 0.01
    ), f"Round trip y mismatch: {back_720.y} != {original_720.y}"
    print("✓ Round-trip 720p -> 4K -> 720p successful")

    print("\n✅ TEST 3 PASSED: Round-trip conversions preserve coordinates\n")
except Exception as e:
    print(f"\n❌ TEST 3 FAILED: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Constants verification
print("TEST 4: 4K Constants Verification")
print("-" * 80)
try:
    from core.constants_4k import (
        BALL_DIAMETER_4K,
        BALL_RADIUS_4K,
        CANONICAL_HEIGHT,
        CANONICAL_WIDTH,
        TABLE_CENTER_4K,
        TABLE_HEIGHT_4K,
        TABLE_WIDTH_4K,
    )

    print(f"Canonical resolution: {CANONICAL_WIDTH}x{CANONICAL_HEIGHT}")
    assert CANONICAL_WIDTH == 3840, f"Expected width 3840, got {CANONICAL_WIDTH}"
    assert CANONICAL_HEIGHT == 2160, f"Expected height 2160, got {CANONICAL_HEIGHT}"
    print("✓ Canonical resolution is 4K (3840x2160)")

    print(f"\nBall dimensions: radius={BALL_RADIUS_4K}, diameter={BALL_DIAMETER_4K}")
    assert 2 * BALL_RADIUS_4K == BALL_DIAMETER_4K, "Ball diameter should be 2x radius"
    print("✓ Ball dimensions consistent")

    print(f"\nTable dimensions: {TABLE_WIDTH_4K}x{TABLE_HEIGHT_4K} (pixels)")
    print(f"Table center: ({TABLE_CENTER_4K[0]}, {TABLE_CENTER_4K[1]})")
    assert (
        TABLE_CENTER_4K[0] == CANONICAL_WIDTH / 2
    ), "Table should be centered horizontally"
    assert (
        TABLE_CENTER_4K[1] == CANONICAL_HEIGHT / 2
    ), "Table should be centered vertically"
    print("✓ Table properly centered in frame")

    print("\n✅ TEST 4 PASSED: 4K constants are correctly defined\n")
except Exception as e:
    print(f"\n❌ TEST 4 FAILED: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Vector math preserves scale
print("TEST 5: Vector Operations Preserve Scale Metadata")
print("-" * 80)
try:
    v1 = Vector2D.from_resolution(100, 100, (1920, 1080))
    v2 = Vector2D.from_resolution(200, 200, (1920, 1080))

    print(f"v1: ({v1.x}, {v1.y}) scale={v1.scale}")
    print(f"v2: ({v2.x}, {v2.y}) scale={v2.scale}")

    # Addition
    v_add = v1 + v2
    print(f"\nv1 + v2: ({v_add.x}, {v_add.y}) scale={v_add.scale}")
    assert v_add.scale == v1.scale, "Addition should preserve scale"
    print("✓ Addition preserves scale")

    # Subtraction
    v_sub = v2 - v1
    print(f"v2 - v1: ({v_sub.x}, {v_sub.y}) scale={v_sub.scale}")
    assert v_sub.scale == v1.scale, "Subtraction should preserve scale"
    print("✓ Subtraction preserves scale")

    # Scalar multiplication
    v_mul = v1 * 2.0
    print(f"v1 * 2: ({v_mul.x}, {v_mul.y}) scale={v_mul.scale}")
    assert v_mul.scale == v1.scale, "Scalar multiplication should preserve scale"
    print("✓ Scalar multiplication preserves scale")

    # Normalization
    v_norm = v1.normalize()
    print(f"normalize(v1): ({v_norm.x}, {v_norm.y}) scale={v_norm.scale}")
    assert v_norm.scale == v1.scale, "Normalization should preserve scale"
    print("✓ Normalization preserves scale")

    print("\n✅ TEST 5 PASSED: Vector operations preserve scale metadata\n")
except Exception as e:
    print(f"\n❌ TEST 5 FAILED: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Final summary
print("=" * 80)
print("ALL TESTS PASSED! ✅")
print("=" * 80)
print()
print("The 4K coordinate system migration is working correctly:")
print("  ✓ Vector2D factory methods create proper scale metadata")
print("  ✓ BallState handles 4K coordinates correctly")
print("  ✓ Round-trip conversions preserve accuracy")
print("  ✓ 4K constants are properly defined")
print("  ✓ Vector operations preserve scale metadata")
print()
print("The system is ready for production use.")
print()
