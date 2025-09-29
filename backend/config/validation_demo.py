#!/usr/bin/env python3
"""
Validation System Demo

This script demonstrates the comprehensive validation system functionality
including ValidationRules, TypeChecker, and ConfigDiffer utilities.
"""

import json
from pathlib import Path
from datetime import datetime

from backend.config.validator.rules import ValidationRules, ValidationError
from backend.config.validator.types import TypeChecker, TypeCheckError
from backend.config.utils.differ import ConfigDiffer, ChangeType
from backend.config.models.schemas import (
    ApplicationConfig,
    VisionConfig,
    CameraSettings,
    create_development_config,
    create_production_config
)


def demo_validation_rules():
    """Demonstrate ValidationRules functionality."""
    print("=" * 60)
    print("VALIDATION RULES DEMO")
    print("=" * 60)

    # Create validator instances
    validator = ValidationRules()
    strict_validator = ValidationRules(strict_mode=True)
    auto_correct_validator = ValidationRules(auto_correct=True)

    print("\n1. Range Validation")
    print("-" * 30)

    # Valid range tests
    print(f"✓ FPS 30 in range [15, 120]: {validator.check_range(30, 15, 120, 'fps')}")
    print(f"✓ Gain 1.5 in range [0.0, 10.0]: {validator.check_range(1.5, 0.0, 10.0, 'gain')}")

    # Invalid range tests
    print(f"✗ FPS 200 in range [15, 120]: {validator.check_range(200, 15, 120, 'fps')}")
    print(f"✗ Device ID -1 in range [0, 10]: {validator.check_range(-1, 0, 10, 'device_id')}")

    # Auto-correction demo
    print(f"\n2. Auto-Correction")
    print("-" * 30)
    corrected_fps = auto_correct_validator.check_range("30", 15, 120, "fps")
    print(f"String '30' auto-corrected to: {corrected_fps}")

    corrected_gain = auto_correct_validator.check_range(15.0, 0.0, 10.0, "gain")
    print(f"Gain 15.0 (above max) auto-corrected to: {corrected_gain}")

    print(f"\n3. Type Validation")
    print("-" * 30)

    # Valid type tests
    print(f"✓ 'hello' is str: {validator.check_type('hello', str, 'name')}")
    print(f"✓ 42 is int: {validator.check_type(42, int, 'count')}")
    print(f"✓ [1,2,3] is list: {validator.check_type([1,2,3], list, 'items')}")

    # Invalid type tests
    print(f"✗ 'hello' is int: {validator.check_type('hello', int, 'count')}")
    print(f"✗ 42 is str: {validator.check_type(42, str, 'name')}")

    print(f"\n4. Pattern Validation")
    print("-" * 30)

    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    print(f"✓ 'test@example.com' matches email pattern: {validator.validate_pattern('test@example.com', email_pattern, 'email')}")
    print(f"✗ 'invalid-email' matches email pattern: {validator.validate_pattern('invalid-email', email_pattern, 'email')}")

    print(f"\n5. Nested Validation")
    print("-" * 30)

    camera_config = {
        "device_id": 0,
        "fps": 30,
        "resolution": (1920, 1080),
        "gain": 1.0
    }

    camera_schema = {
        "device_id": {"type": "int", "min": 0, "max": 10, "required": True},
        "fps": {"type": "int", "min": 15, "max": 120, "required": True},
        "resolution": {"type": "tuple", "required": True},
        "gain": {"type": "float", "min": 0.0, "max": 10.0, "required": False}
    }

    result = validator.validate_nested(camera_config, camera_schema, "camera")
    print(f"✓ Camera config validation: {result}")

    # Show validation errors
    if validator.get_validation_errors():
        print(f"\nValidation Errors:")
        for error in validator.get_validation_errors():
            print(f"  - {error}")


def demo_type_checker():
    """Demonstrate TypeChecker functionality."""
    print("\n\n" + "=" * 60)
    print("TYPE CHECKER DEMO")
    print("=" * 60)

    checker = TypeChecker()
    strict_checker = TypeChecker(strict_mode=True)

    print("\n1. Basic Type Checking")
    print("-" * 30)

    print(f"✓ 'hello' is str: {checker.check('hello', str)}")
    print(f"✓ 42 is int: {checker.check(42, int)}")
    print(f"✓ 3.14 is float: {checker.check(3.14, float)}")
    print(f"✗ 'hello' is int: {checker.check('hello', int)}")

    print(f"\n2. Generic Type Checking")
    print("-" * 30)

    from typing import List, Dict, Optional, Union

    print(f"✓ ['a','b','c'] is List[str]: {checker.check(['a','b','c'], List[str])}")
    print(f"✓ {{'a':1,'b':2}} is Dict[str,int]: {checker.check({'a':1,'b':2}, Dict[str,int])}")
    print(f"✓ 'hello' is Optional[str]: {checker.check('hello', Optional[str])}")
    print(f"✓ None is Optional[str]: {checker.check(None, Optional[str])}")
    print(f"✓ 42 is Union[str,int]: {checker.check(42, Union[str,int])}")

    print(f"\n3. Pydantic Model Checking")
    print("-" * 30)

    camera_dict = {
        "device_id": 0,
        "resolution": (1920, 1080),
        "fps": 30,
        "gain": 1.0,
        "brightness": 0.5
    }

    print(f"✓ Camera dict validates against CameraSettings: {checker.check(camera_dict, CameraSettings)}")

    # Invalid camera config
    invalid_camera = {"device_id": "not_int", "fps": -10}
    print(f"✗ Invalid camera dict validates: {checker.check(invalid_camera, CameraSettings)}")

    print(f"\n4. Type Coercion")
    print("-" * 30)

    coercion_checker = TypeChecker(allow_coercion=True)
    no_coercion_checker = TypeChecker(allow_coercion=False)

    print(f"✓ '42' coerced to int: {coercion_checker.check('42', int)}")
    print(f"✗ '42' without coercion: {no_coercion_checker.check('42', int)}")
    print(f"✓ 'true' coerced to bool: {coercion_checker.check('true', bool)}")
    print(f"✓ '/path/to/file' coerced to Path: {coercion_checker.check('/path/to/file', Path)}")


def demo_config_differ():
    """Demonstrate ConfigDiffer functionality."""
    print("\n\n" + "=" * 60)
    print("CONFIG DIFFER DEMO")
    print("=" * 60)

    differ = ConfigDiffer()

    print("\n1. Basic Configuration Diffing")
    print("-" * 30)

    # Original config
    config_v1 = {
        "vision": {
            "camera": {
                "device_id": 0,
                "fps": 30,
                "resolution": (1920, 1080)
            },
            "detection": {
                "min_ball_radius": 10,
                "max_ball_radius": 40
            }
        },
        "core": {
            "physics": {
                "gravity": 9.81,
                "friction": 0.01
            }
        }
    }

    # Updated config
    config_v2 = {
        "vision": {
            "camera": {
                "device_id": 1,  # Changed
                "fps": 60,       # Changed
                "resolution": (1920, 1080)  # Unchanged
            },
            "detection": {
                "min_ball_radius": 15,     # Changed
                "max_ball_radius": 40,     # Unchanged
                "sensitivity": 0.8         # Added
            }
        },
        "core": {
            "physics": {
                "gravity": 9.81,     # Unchanged
                "friction": 0.02,    # Changed
                "air_resistance": 0.001  # Added
            }
        }
    }

    diff_result = differ.diff(config_v1, config_v2)

    print(f"Total changes: {diff_result['summary'].total_changes}")
    print(f"Added: {diff_result['summary'].added_count}")
    print(f"Modified: {diff_result['summary'].modified_count}")
    print(f"Removed: {diff_result['summary'].removed_count}")

    print(f"\n2. Detailed Changes")
    print("-" * 30)

    for change in diff_result['changes']:
        symbol = {
            ChangeType.ADDED: "+",
            ChangeType.REMOVED: "-",
            ChangeType.MODIFIED: "~"
        }.get(change.change_type, "?")

        print(f"{symbol} {change.path}")
        if change.change_type == ChangeType.ADDED:
            print(f"    Added: {change.new_value}")
        elif change.change_type == ChangeType.REMOVED:
            print(f"    Removed: {change.old_value}")
        elif change.change_type == ChangeType.MODIFIED:
            print(f"    {change.old_value} → {change.new_value}")

    print(f"\n3. Diff Report")
    print("-" * 30)

    report = differ.format_diff_report(diff_result)
    print(report[:500] + "..." if len(report) > 500 else report)

    print(f"\n4. Version Comparison")
    print("-" * 30)

    # Multiple versions
    config_v3 = {
        **config_v2,
        "api": {
            "port": 8000,
            "enable_docs": True
        }
    }

    versions = [
        ("v1.0", config_v1),
        ("v1.1", config_v2),
        ("v1.2", config_v3)
    ]

    comparisons = differ.compare_versions(versions)

    for comparison_name, comparison_result in comparisons.items():
        print(f"{comparison_name}: {comparison_result['summary'].total_changes} changes")


def demo_real_world_scenario():
    """Demonstrate real-world validation scenario."""
    print("\n\n" + "=" * 60)
    print("REAL-WORLD SCENARIO DEMO")
    print("=" * 60)

    # Create configuration instances
    validator = ValidationRules()
    type_checker = TypeChecker()
    differ = ConfigDiffer()

    print("\n1. Configuration Loading and Validation")
    print("-" * 50)

    # Load development configuration
    dev_config = create_development_config()
    prod_config = create_production_config()

    # Convert to dictionaries for manipulation
    dev_dict = dev_config.model_dump()
    prod_dict = prod_config.model_dump()

    print(f"✓ Development config loaded and validated")
    print(f"✓ Production config loaded and validated")

    # Validate against schemas
    dev_valid = type_checker.check(dev_dict, ApplicationConfig)
    prod_valid = type_checker.check(prod_dict, ApplicationConfig)

    print(f"✓ Dev config schema validation: {dev_valid}")
    print(f"✓ Prod config schema validation: {prod_valid}")

    print(f"\n2. Configuration Migration Simulation")
    print("-" * 50)

    # Simulate configuration migration
    migration_diff = differ.diff(dev_dict, prod_dict)

    print(f"Migration changes: {migration_diff['summary'].total_changes}")
    print(f"Settings to add: {migration_diff['summary'].added_count}")
    print(f"Settings to modify: {migration_diff['summary'].modified_count}")
    print(f"Settings to remove: {migration_diff['summary'].removed_count}")

    # Show critical changes
    critical_changes = [
        change for change in migration_diff['changes']
        if any(keyword in change.path.lower() for keyword in ['debug', 'ssl', 'port', 'auth'])
    ]

    if critical_changes:
        print(f"\nCritical changes requiring attention:")
        for change in critical_changes:
            print(f"  {change.change_type.value}: {change.path}")

    print(f"\n3. Validation Error Recovery")
    print("-" * 50)

    # Simulate invalid configuration
    invalid_config = {
        "vision": {
            "camera": {
                "device_id": -1,       # Invalid: below minimum
                "fps": "sixty",        # Invalid: wrong type
                "resolution": "1920x1080",  # Invalid: wrong format
                "gain": 15.0           # Invalid: above maximum
            }
        }
    }

    print("Attempting to validate invalid configuration...")

    # Collect validation errors
    validator.clear_validation_errors()

    # Validate camera settings
    camera_config = invalid_config["vision"]["camera"]

    validator.check_range(camera_config["device_id"], 0, 10, "vision.camera.device_id")
    validator.check_type(camera_config["fps"], int, "vision.camera.fps")
    validator.check_type(camera_config["resolution"], tuple, "vision.camera.resolution")
    validator.check_range(camera_config["gain"], 0.0, 10.0, "vision.camera.gain")

    errors = validator.get_validation_errors()
    print(f"Found {len(errors)} validation errors:")
    for error in errors:
        print(f"  - {error}")

    # Demonstrate auto-correction
    print(f"\n4. Auto-Correction")
    print("-" * 50)

    auto_corrector = ValidationRules(auto_correct=True)

    print("Attempting auto-correction...")
    corrected_device_id = auto_corrector.check_range(-1, 0, 10, "device_id")
    corrected_gain = auto_corrector.check_range(15.0, 0.0, 10.0, "gain")

    print(f"device_id -1 → {corrected_device_id}")
    print(f"gain 15.0 → {corrected_gain}")

    print(f"\n5. Configuration Backup and Rollback Simulation")
    print("-" * 50)

    # Simulate backup creation
    backup_config = dev_dict.copy()

    # Make changes
    modified_config = dev_dict.copy()
    modified_config["vision"]["camera"]["fps"] = 60
    modified_config["vision"]["debug"] = True
    modified_config["system"]["debug"] = False  # Rollback this change

    # Calculate rollback diff
    rollback_diff = differ.diff(modified_config, backup_config)

    print(f"Changes to rollback: {rollback_diff['summary'].total_changes}")
    print("Rollback operations:")
    for change in rollback_diff['changes']:
        if change.change_type == ChangeType.MODIFIED:
            print(f"  Restore {change.path}: {change.new_value} ← {change.old_value}")
        elif change.change_type == ChangeType.ADDED:
            print(f"  Remove {change.path}")
        elif change.change_type == ChangeType.REMOVED:
            print(f"  Add {change.path}: {change.old_value}")


def main():
    """Run all validation demos."""
    print("BILLIARDS TRAINER - CONFIGURATION VALIDATION SYSTEM DEMO")
    print("=" * 80)
    print(f"Demo started at: {datetime.now().isoformat()}")

    try:
        demo_validation_rules()
        demo_type_checker()
        demo_config_differ()
        demo_real_world_scenario()

        print("\n\n" + "=" * 80)
        print("✓ ALL VALIDATION DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except Exception as e:
        print(f"\n\n✗ DEMO FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())