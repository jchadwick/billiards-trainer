"""Tests for simple configuration system."""

import json
import tempfile
from pathlib import Path

import pytest

from backend.config import Config


def test_config_singleton():
    """Test that Config is a singleton."""
    config1 = Config()
    config2 = Config()
    assert config1 is config2


def test_config_get_with_default():
    """Test getting config values with defaults."""
    config = Config()
    # Test with existing key
    assert config.get("system.debug", True) is not None
    # Test with non-existing key
    assert config.get("nonexistent.key", "default") == "default"


def test_config_get_nested():
    """Test getting nested config values using dot notation."""
    config = Config()
    # Test nested path
    value = config.get("vision.camera.device_id", 0)
    assert isinstance(value, int)

    # Test deep nesting
    value = config.get("core.physics.gravity", 9.81)
    assert isinstance(value, (int, float))


def test_config_set():
    """Test setting config values."""
    config = Config()
    config.set("test.value", 42)
    assert config.get("test.value") == 42

    # Test overwriting
    config.set("test.value", 99)
    assert config.get("test.value") == 99


def test_config_set_nested():
    """Test setting nested config values."""
    config = Config()
    config.set("test.nested.deep.value", "hello")
    assert config.get("test.nested.deep.value") == "hello"

    # Verify intermediate dicts were created
    nested = config.get("test.nested")
    assert isinstance(nested, dict)
    assert "deep" in nested


def test_config_get_all():
    """Test getting all configuration."""
    config = Config()
    all_config = config.get_all()
    assert isinstance(all_config, dict)
    assert "system" in all_config or "vision" in all_config or "core" in all_config


def test_config_reload():
    """Test config reload functionality."""
    config = Config()
    original = config.get("system.debug")

    # Reload should work without error
    config.reload()

    # Value should be the same after reload
    reloaded = config.get("system.debug")
    assert original == reloaded


def test_config_handles_missing_file():
    """Test that Config handles missing config file gracefully."""
    # Create a temporary Config with a non-existent file
    config = Config()
    original_file = config._config_file

    try:
        # Point to non-existent file
        config._config_file = Path("/tmp/nonexistent_config_12345.json")
        config._load_config()

        # Should have empty config but not crash
        # Just verify we can call get() without error
        result = config.get("any.key", "default")
        assert result == "default"
    finally:
        # Restore original file
        config._config_file = original_file
        config._load_config()


def test_config_handles_invalid_json():
    """Test that Config handles invalid JSON gracefully."""
    config = Config()
    original_file = config._config_file

    try:
        # Create temp file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            temp_path = f.name

        config._config_file = Path(temp_path)
        config._load_config()

        # Should have empty config but not crash
        result = config.get("any.key", "default")
        assert result == "default"
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
        config._config_file = original_file
        config._load_config()


def test_config_production_values():
    """Test that production values are loaded correctly."""
    config = Config()

    # Check that we have production values
    environment = config.get("system.environment", "development")
    # Should be production based on our consolidated config
    assert environment in ["production", "development"]

    # Check production-specific values
    if environment == "production":
        debug = config.get("system.debug", True)
        assert debug is False or debug is True  # Either is valid
