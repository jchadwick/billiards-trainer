#!/usr/bin/env python3
"""Test script for calibration storage functionality."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test 1: Database initialization
print("=" * 60)
print("TEST 1: Database Initialization")
print("=" * 60)

try:
    # Import models FIRST so they register with Base
    from backend.api.database import SessionLocal, init_db
    from backend.api.models.calibration_db import TableCalibration

    # Now initialize database (creates tables)
    init_db()
    print("✅ Database initialized successfully")

    # Create a test session
    db = SessionLocal()
    print("✅ Database session created")
    db.close()
except Exception as e:
    print(f"❌ Database initialization failed: {e}")
    sys.exit(1)

# Test 2: Model creation and persistence
print("\n" + "=" * 60)
print("TEST 2: Model Creation and Persistence")
print("=" * 60)

try:
    from backend.api.models.calibration_db import (
        get_calibration_by_session_id,
        get_default_calibration,
        get_latest_calibration,
        list_calibrations,
        save_calibration,
    )

    # Create test calibration data
    test_session_id = f"test_cal_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    test_points = [
        {
            "point_id": "p1",
            "screen_x": 100.0,
            "screen_y": 100.0,
            "world_x": 0.0,
            "world_y": 0.0,
            "confidence": 0.95,
        },
        {
            "point_id": "p2",
            "screen_x": 1820.0,
            "screen_y": 100.0,
            "world_x": 2.54,
            "world_y": 0.0,
            "confidence": 0.92,
        },
        {
            "point_id": "p3",
            "screen_x": 1820.0,
            "screen_y": 980.0,
            "world_x": 2.54,
            "world_y": 1.27,
            "confidence": 0.93,
        },
        {
            "point_id": "p4",
            "screen_x": 100.0,
            "screen_y": 980.0,
            "world_x": 0.0,
            "world_y": 1.27,
            "confidence": 0.94,
        },
    ]

    test_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    # Create calibration model
    db = SessionLocal()
    try:
        calibration = TableCalibration(
            session_id=test_session_id,
            calibration_type="standard",
            status="applied",
            created_at=datetime.now(timezone.utc),
            applied_at=datetime.now(timezone.utc),
            points_required=4,
            points_captured=4,
            points=test_points,
            transformation_matrix=test_matrix,
            metadata={"test": True},
            accuracy=0.935,
            max_error=0.5,
            mean_error=0.3,
            rms_error=0.35,
            is_default=True,
            created_by="test_script",
            applied_by="test_script",
            notes="Test calibration data",
        )

        # Save to database
        saved = save_calibration(db, calibration)
        print(f"✅ Calibration saved with ID: {saved.id}")
        print(f"   Session ID: {saved.session_id}")
        print(f"   Accuracy: {saved.accuracy}")
        print(f"   Points: {saved.points_captured}/{saved.points_required}")

    finally:
        db.close()

except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Data retrieval
print("\n" + "=" * 60)
print("TEST 3: Data Retrieval")
print("=" * 60)

try:
    db = SessionLocal()
    try:
        # Retrieve by session ID
        retrieved = get_calibration_by_session_id(db, test_session_id)
        if retrieved:
            print("✅ Retrieved calibration by session ID:")
            print(f"   ID: {retrieved.id}")
            print(f"   Type: {retrieved.calibration_type}")
            print(f"   Status: {retrieved.status}")
            print(f"   Accuracy: {retrieved.accuracy}")
        else:
            print("❌ Could not retrieve calibration by session ID")

        # Get latest calibration
        latest = get_latest_calibration(db)
        if latest:
            print("✅ Latest calibration:")
            print(f"   Session ID: {latest.session_id}")
            print(f"   Applied: {latest.applied_at}")
        else:
            print("ℹ️  No applied calibrations found")

        # Get default calibration
        default = get_default_calibration(db)
        if default:
            print("✅ Default calibration:")
            print(f"   Session ID: {default.session_id}")
            print(f"   Is default: {default.is_default}")
        else:
            print("ℹ️  No default calibration set")

        # List all calibrations
        all_cals = list_calibrations(db, limit=10)
        print(f"✅ Total calibrations in database: {len(all_cals)}")

    finally:
        db.close()

except Exception as e:
    print(f"❌ Data retrieval failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify data structure
print("\n" + "=" * 60)
print("TEST 4: Data Structure Verification")
print("=" * 60)

try:
    db = SessionLocal()
    try:
        retrieved = get_calibration_by_session_id(db, test_session_id)

        # Verify transformation matrix
        if retrieved.transformation_matrix:
            assert len(retrieved.transformation_matrix) == 3
            assert all(len(row) == 3 for row in retrieved.transformation_matrix)
            print("✅ Transformation matrix structure valid (3x3)")
        else:
            print("⚠️  No transformation matrix found")

        # Verify points
        if retrieved.points:
            assert len(retrieved.points) == 4
            assert all("screen_x" in p for p in retrieved.points)
            assert all("screen_y" in p for p in retrieved.points)
            assert all("world_x" in p for p in retrieved.points)
            assert all("world_y" in p for p in retrieved.points)
            print(f"✅ All {len(retrieved.points)} calibration points valid")
        else:
            print("❌ No calibration points found")

        # Test to_dict() method
        cal_dict = retrieved.to_dict()
        assert "id" in cal_dict
        assert "session_id" in cal_dict
        assert "transformation_matrix" in cal_dict
        assert "points" in cal_dict
        print("✅ to_dict() method working correctly")

        # Print sample calibration data
        print("\nSample calibration data:")
        print(f"  Session ID: {cal_dict['session_id']}")
        print(f"  Type: {cal_dict['calibration_type']}")
        print(f"  Status: {cal_dict['status']}")
        print(f"  Accuracy: {cal_dict['accuracy']:.3f}")
        print(f"  Points captured: {cal_dict['points_captured']}")
        print(
            f"  Transformation matrix shape: {len(cal_dict['transformation_matrix'])}x{len(cal_dict['transformation_matrix'][0])}"
        )

    finally:
        db.close()

except Exception as e:
    print(f"❌ Data structure verification failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✅")
print("=" * 60)
print("\nCalibration storage functionality is working correctly.")
print(f"Database location: {Path(__file__).parent / 'data' / 'billiards_trainer.db'}")
print("\nYou can now use the following API endpoints:")
print("  POST   /api/v1/vision/calibration/storage/save")
print("  GET    /api/v1/vision/calibration/storage/retrieve/{identifier}")
print("  GET    /api/v1/vision/calibration/storage/default")
print("  GET    /api/v1/vision/calibration/storage/latest")
print("  GET    /api/v1/vision/calibration/storage/list")
print("  PUT    /api/v1/vision/calibration/storage/update/{session_id}")
print("  POST   /api/v1/vision/calibration/storage/set-default/{session_id}")
print("  DELETE /api/v1/vision/calibration/storage/delete/{session_id}")
