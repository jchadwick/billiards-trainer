"""Basic functionality test for configuration loading system.

This test validates that the core functionality works correctly
for the main requirements FR-CFG-001 through FR-CFG-005.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from ..loader.env import EnvironmentLoader
from ..loader.file import FileLoader
from ..loader.merger import ConfigSource, ConfigurationMerger


def test_basic_file_loading():
    """Test basic file loading functionality (FR-CFG-001, FR-CFG-002)."""
    # Create a test configuration
    test_config = {
        "app": {"name": "billiards-trainer", "debug": True},
        "database": {"host": "localhost", "port": 5432},
    }

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_config, f)
        temp_path = f.name

    try:
        loader = FileLoader()
        result = loader.load_file(temp_path)

        assert result["app"]["name"] == "billiards-trainer"
        assert result["app"]["debug"] is True
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == 5432

    finally:
        os.unlink(temp_path)


def test_basic_environment_loading():
    """Test basic environment variable loading (FR-CFG-001)."""
    original_env = os.environ.copy()

    try:
        # Set test environment variables
        os.environ["TEST_APP_NAME"] = "test-app"
        os.environ["TEST_APP_DEBUG"] = "true"
        os.environ["TEST_DATABASE_PORT"] = "3306"

        loader = EnvironmentLoader(prefix="TEST_")
        result = loader.load_environment()

        assert result["app"]["name"] == "test-app"
        assert result["app"]["debug"] is True
        assert result["database"]["port"] == 3306

    finally:
        # Restore environment
        os.environ.clear()
        os.environ.update(original_env)


def test_basic_configuration_merging():
    """Test basic configuration merging with precedence (FR-CFG-003)."""
    config1 = {
        "app": {"name": "base", "version": "1.0"},
        "database": {"host": "localhost"},
    }

    config2 = {"app": {"name": "override", "debug": True}, "database": {"port": 5432}}

    merger = ConfigurationMerger()
    result = merger.merge_configurations(
        [config1, config2], [ConfigSource.FILE, ConfigSource.ENVIRONMENT]
    )

    # Environment should override file
    assert result["app"]["name"] == "override"
    assert result["app"]["version"] == "1.0"  # Should be preserved
    assert result["app"]["debug"] is True
    assert result["database"]["host"] == "localhost"
    assert result["database"]["port"] == 5432


def test_default_values():
    """Test default value support (FR-CFG-004)."""
    defaults = {
        "app": {"name": "default-app", "debug": False},
        "database": {"host": "localhost", "port": 5432},
    }

    override_config = {"app": {"debug": True}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(override_config, f)
        temp_path = f.name

    try:
        loader = FileLoader(default_values=defaults)
        result = loader.load_with_defaults(temp_path)

        # Should have default values
        assert result["app"]["name"] == "default-app"
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == 5432

        # Should have override values
        assert result["app"]["debug"] is True

    finally:
        os.unlink(temp_path)


def test_configuration_inheritance():
    """Test configuration inheritance (FR-CFG-005)."""
    parent_config = {
        "app": {"name": "parent-app", "version": "1.0"},
        "database": {"host": "localhost"},
    }

    child_config = {
        "inherit": "parent.json",
        "app": {"name": "child-app", "debug": True},
        "database": {"port": 5432},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        parent_path = Path(temp_dir) / "parent.json"
        child_path = Path(temp_dir) / "child.json"

        with open(parent_path, "w") as f:
            json.dump(parent_config, f)

        with open(child_path, "w") as f:
            json.dump(child_config, f)

        loader = FileLoader()
        result = loader.load_with_inheritance(child_path)

        # Should inherit from parent
        assert result["app"]["version"] == "1.0"
        assert result["database"]["host"] == "localhost"

        # Should override parent values
        assert result["app"]["name"] == "child-app"

        # Should have child-specific values
        assert result["app"]["debug"] is True
        assert result["database"]["port"] == 5432


def test_multiple_source_integration():
    """Test integration of multiple configuration sources."""
    # Default values
    defaults = {
        "app": {"name": "default", "debug": False, "version": "1.0"},
        "database": {"host": "localhost", "port": 5432},
    }

    # File configuration
    file_config = {"app": {"debug": True}, "database": {"host": "file-host"}}

    # Save original environment
    original_env = os.environ.copy()

    try:
        # Set environment variables
        os.environ["TEST_APP_DEBUG"] = "false"  # Override file
        os.environ["TEST_DATABASE_PORT"] = "3306"  # Override default

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(file_config, f)
            temp_path = f.name

        # Load configurations
        file_loader = FileLoader(default_values=defaults)
        env_loader = EnvironmentLoader(prefix="TEST_")
        merger = ConfigurationMerger()

        file_result = file_loader.load_with_defaults(temp_path)
        env_result = env_loader.load_environment()

        # Merge with precedence: file < environment
        final_result = merger.merge_configurations(
            [file_result, env_result], [ConfigSource.FILE, ConfigSource.ENVIRONMENT]
        )

        # Verify final configuration
        assert final_result["app"]["name"] == "default"  # from defaults
        assert final_result["app"]["debug"] is False  # env overrides file
        assert final_result["app"]["version"] == "1.0"  # from defaults
        assert final_result["database"]["host"] == "file-host"  # from file
        assert final_result["database"]["port"] == 3306  # from environment

    finally:
        # Cleanup
        os.unlink(temp_path)
        os.environ.clear()
        os.environ.update(original_env)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
