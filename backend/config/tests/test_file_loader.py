"""Tests for file configuration loader.

Tests all functionality including JSON/YAML/INI loading, inheritance,
defaults, and error handling.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from backend.config.loader.file import (
    ConfigFormat,
    FileLoader,
    FileLoadError,
    FormatError,
)


class TestFileLoader:
    """Test cases for FileLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.default_values = {
            "app": {"name": "test-app", "debug": False},
            "database": {"host": "localhost", "port": 5432},
        }
        self.loader = FileLoader(default_values=self.default_values)

    def test_init_with_defaults(self):
        """Test FileLoader initialization with default values."""
        assert self.loader.default_values == self.default_values
        assert self.loader.encoding == "utf-8"
        assert self.loader._loaded_files == {}

    def test_init_without_defaults(self):
        """Test FileLoader initialization without default values."""
        loader = FileLoader()
        assert loader.default_values == {}

    def test_detect_format_json(self):
        """Test auto-detection of JSON format."""
        format = self.loader._detect_format(Path("config.json"))
        assert format == ConfigFormat.JSON

    def test_detect_format_yaml(self):
        """Test auto-detection of YAML format."""
        format = self.loader._detect_format(Path("config.yaml"))
        assert format == ConfigFormat.YAML

        format = self.loader._detect_format(Path("config.yml"))
        assert format == ConfigFormat.YML

    def test_detect_format_ini(self):
        """Test auto-detection of INI format."""
        format = self.loader._detect_format(Path("config.ini"))
        assert format == ConfigFormat.INI

    def test_detect_format_unsupported(self):
        """Test detection of unsupported format."""
        with pytest.raises(FormatError):
            self.loader._detect_format(Path("config.unknown"))

    def test_parse_json_content(self):
        """Test parsing JSON content."""
        content = '{"key": "value", "number": 42}'
        result = self.loader._parse_content(content, ConfigFormat.JSON)
        assert result == {"key": "value", "number": 42}

    def test_parse_yaml_content(self):
        """Test parsing YAML content."""
        content = """
        key: value
        number: 42
        nested:
          inner: true
        """
        result = self.loader._parse_content(content, ConfigFormat.YAML)
        assert result == {"key": "value", "number": 42, "nested": {"inner": True}}

    def test_parse_ini_content(self):
        """Test parsing INI content."""
        content = """
        [section1]
        key1 = value1
        key2 = value2

        [section2]
        key3 = value3
        """
        result = self.loader._parse_content(content, ConfigFormat.INI)
        assert "section1" in result
        assert result["section1"]["key1"] == "value1"
        assert result["section1"]["key2"] == "value2"
        assert result["section2"]["key3"] == "value3"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON content."""
        content = '{"invalid": json}'
        with pytest.raises(FormatError):
            self.loader._parse_content(content, ConfigFormat.JSON)

    def test_deep_merge(self):
        """Test deep merging of dictionaries."""
        base = {
            "app": {"name": "base", "version": "1.0"},
            "database": {"host": "localhost"},
        }
        override = {
            "app": {"name": "override", "debug": True},
            "cache": {"type": "redis"},
        }

        result = self.loader._deep_merge(base, override)

        assert result["app"]["name"] == "override"
        assert result["app"]["version"] == "1.0"
        assert result["app"]["debug"] is True
        assert result["database"]["host"] == "localhost"
        assert result["cache"]["type"] == "redis"

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        result = self.loader.load_file("/nonexistent/path.json")
        assert result == {}

    def test_load_file_with_temp_json(self):
        """Test loading a temporary JSON file."""
        config_data = {"app": {"name": "test", "debug": True}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = self.loader.load_file(temp_path)
            assert result == config_data
            assert temp_path in self.loader._loaded_files
        finally:
            os.unlink(temp_path)

    def test_load_file_with_temp_yaml(self):
        """Test loading a temporary YAML file."""
        config_data = {"app": {"name": "test", "debug": True}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            result = self.loader.load_file(temp_path)
            assert result == config_data
        finally:
            os.unlink(temp_path)

    def test_load_multiple_files(self):
        """Test loading multiple configuration files."""
        config1 = {"key1": "value1"}
        config2 = {"key2": "value2"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            json.dump(config1, f1)
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            json.dump(config2, f2)
            path2 = f2.name

        try:
            results = self.loader.load_multiple([path1, path2])
            assert len(results) == 2
            assert config1 in results
            assert config2 in results
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_load_multiple_with_missing(self):
        """Test loading multiple files with some missing."""
        config1 = {"key1": "value1"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config1, f)
            path1 = f.name

        try:
            results = self.loader.load_multiple(
                [path1, "/nonexistent.json"], ignore_missing=True
            )
            assert len(results) == 1
            assert results[0] == config1
        finally:
            os.unlink(path1)

    def test_load_multiple_fail_on_missing(self):
        """Test loading multiple files failing on missing."""
        with pytest.raises(FileLoadError):
            self.loader.load_multiple(["/nonexistent.json"], ignore_missing=False)

    def test_load_with_defaults(self):
        """Test loading file with default values applied."""
        config_data = {"app": {"name": "override"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = self.loader.load_with_defaults(temp_path)

            # Should have override value
            assert result["app"]["name"] == "override"
            # Should have default values
            assert result["app"]["debug"] is False
            assert result["database"]["host"] == "localhost"
            assert result["database"]["port"] == 5432
        finally:
            os.unlink(temp_path)

    def test_load_with_inheritance(self):
        """Test loading configuration with inheritance."""
        parent_config = {"app": {"name": "parent", "version": "1.0"}}
        child_config = {
            "inherit": "parent.json",
            "app": {"name": "child", "debug": True},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            parent_path = Path(temp_dir) / "parent.json"
            child_path = Path(temp_dir) / "child.json"

            with open(parent_path, "w") as f:
                json.dump(parent_config, f)

            with open(child_path, "w") as f:
                json.dump(child_config, f)

            result = self.loader.load_with_inheritance(child_path)

            # Child should override parent
            assert result["app"]["name"] == "child"
            assert result["app"]["debug"] is True
            # Should inherit from parent
            assert result["app"]["version"] == "1.0"

    def test_is_file_modified(self):
        """Test file modification detection."""
        config_data = {"key": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # File not loaded yet
            assert self.loader.is_file_modified(temp_path) is True

            # Load file
            self.loader.load_file(temp_path)
            assert self.loader.is_file_modified(temp_path) is False

            # Modify file
            import time

            time.sleep(0.1)  # Ensure different mtime
            with open(temp_path, "w") as f:
                json.dump({"key": "new_value"}, f)

            assert self.loader.is_file_modified(temp_path) is True
        finally:
            os.unlink(temp_path)

    def test_clear_cache(self):
        """Test clearing the file cache."""
        config_data = {"key": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            self.loader.load_file(temp_path)
            assert len(self.loader._loaded_files) == 1

            self.loader.clear_cache()
            assert len(self.loader._loaded_files) == 0
        finally:
            os.unlink(temp_path)

    def test_load_directory_instead_of_file(self):
        """Test loading a directory instead of a file."""
        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(FileLoadError):
            self.loader.load_file(temp_dir)


class TestConfigFormatEnum:
    """Test cases for ConfigFormat enum."""

    def test_format_values(self):
        """Test format enum values."""
        assert ConfigFormat.JSON.value == "json"
        assert ConfigFormat.YAML.value == "yaml"
        assert ConfigFormat.YML.value == "yml"
        assert ConfigFormat.INI.value == "ini"
        assert ConfigFormat.TOML.value == "toml"


class TestFileLoaderIntegration:
    """Integration tests for FileLoader."""

    def test_complex_configuration_loading(self):
        """Test loading a complex configuration with multiple features."""
        # Create a complex configuration structure
        default_config = {
            "app": {
                "name": "billiards-trainer",
                "version": "1.0.0",
                "debug": False,
                "features": {"vision": True, "projector": True},
            },
            "database": {"host": "localhost", "port": 5432, "name": "billiards"},
        }

        override_config = {
            "app": {"debug": True, "features": {"vision": False, "ai_analysis": True}},
            "database": {"host": "production-db"},
            "logging": {"level": "DEBUG"},
        }

        loader = FileLoader(default_values=default_config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(override_config, f)
            temp_path = f.name

        try:
            result = loader.load_with_defaults(temp_path)

            # Check merged configuration
            assert result["app"]["name"] == "billiards-trainer"  # from default
            assert result["app"]["version"] == "1.0.0"  # from default
            assert result["app"]["debug"] is True  # overridden
            assert result["app"]["features"]["vision"] is False  # overridden
            assert result["app"]["features"]["projector"] is True  # from default
            assert result["app"]["features"]["ai_analysis"] is True  # new
            assert result["database"]["host"] == "production-db"  # overridden
            assert result["database"]["port"] == 5432  # from default
            assert result["logging"]["level"] == "DEBUG"  # new

        finally:
            os.unlink(temp_path)
