"""Integration tests for configuration migrations with ConfigurationModule."""

import json
import tempfile
from pathlib import Path

import pytest

from backend.config.manager import ConfigurationModule


class TestMigrationIntegration:
    """Test migration integration with ConfigurationModule."""

    def test_load_legacy_file_config_auto_migrates(self, tmp_path):
        """Test that loading legacy file configuration triggers migration."""
        # Create a legacy configuration file
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        legacy_config = {
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                    "loop_video": True,
                    "resolution": [1920, 1080],
                }
            }
        }

        config_file = config_dir / "legacy.json"
        with open(config_file, "w") as f:
            json.dump(legacy_config, f)

        # Load configuration through ConfigurationModule
        config_module = ConfigurationModule(
            config_dir=config_dir, enable_hot_reload=False
        )
        config_module.load_config(config_file)

        # Check that migration was applied
        config_module.get_all()

        # Should have video_source_type
        assert (
            config_module.get("vision.camera.video_source_type") == "file"
        ), "Migration should set video_source_type to 'file'"

        # Should have video_file_path
        assert (
            config_module.get("vision.camera.video_file_path") == "/path/to/video.mp4"
        ), "Migration should set video_file_path"

        # Should NOT have string device_id
        device_id = config_module.get("vision.camera.device_id")
        assert device_id is None or isinstance(
            device_id, int
        ), "Migration should remove string device_id"

        # Should preserve loop_video
        assert (
            config_module.get("vision.camera.loop_video") is True
        ), "Migration should preserve loop_video for file sources"

    def test_load_legacy_camera_config_auto_migrates(self, tmp_path):
        """Test that loading legacy camera configuration triggers migration."""
        # Create a legacy configuration file
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        legacy_config = {
            "vision": {
                "camera": {
                    "device_id": 1,
                    "resolution": [1920, 1080],
                    "fps": 30,
                }
            }
        }

        config_file = config_dir / "legacy.json"
        with open(config_file, "w") as f:
            json.dump(legacy_config, f)

        # Load configuration through ConfigurationModule
        config_module = ConfigurationModule(
            config_dir=config_dir, enable_hot_reload=False
        )
        config_module.load_config(config_file)

        # Check that migration was applied
        assert (
            config_module.get("vision.camera.video_source_type") == "camera"
        ), "Migration should set video_source_type to 'camera'"

        # Should preserve device_id
        assert (
            config_module.get("vision.camera.device_id") == 1
        ), "Migration should preserve device_id for camera sources"

        # Should NOT have loop_video
        assert (
            config_module.get("vision.camera.loop_video") is None
        ), "Migration should not add loop_video for camera sources"

    def test_load_modern_config_skips_migration(self, tmp_path):
        """Test that loading modern configuration skips migration."""
        # Create a modern configuration file
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        modern_config = {
            "vision": {
                "camera": {
                    "video_source_type": "file",
                    "video_file_path": "/path/to/video.mp4",
                    "loop_video": True,
                    "resolution": [1920, 1080],
                }
            }
        }

        config_file = config_dir / "modern.json"
        with open(config_file, "w") as f:
            json.dump(modern_config, f)

        # Load configuration through ConfigurationModule
        config_module = ConfigurationModule(
            config_dir=config_dir, enable_hot_reload=False
        )
        config_module.load_config(config_file)

        # Check that configuration is unchanged
        assert (
            config_module.get("vision.camera.video_source_type") == "file"
        ), "Modern config should be unchanged"
        assert (
            config_module.get("vision.camera.video_file_path") == "/path/to/video.mp4"
        ), "Modern config should be unchanged"

    def test_migration_with_missing_camera_section(self, tmp_path):
        """Test migration with configuration missing camera section."""
        # Create a configuration without camera section
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config = {
            "system": {"debug": False},
            "api": {"port": 8000},
        }

        config_file = config_dir / "no_camera.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        # Load configuration through ConfigurationModule
        config_module = ConfigurationModule(
            config_dir=config_dir, enable_hot_reload=False
        )

        # Should not raise an error
        result = config_module.load_config(config_file)
        assert result is True, "Loading config without camera section should succeed"

    def test_migration_with_int_device_id_and_loop_video(self, tmp_path):
        """Test migration handles misconfigured int device_id with loop_video."""
        # Create a misconfigured configuration file
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        misconfigured = {
            "vision": {
                "camera": {
                    "device_id": 0,
                    "loop_video": True,  # Invalid for camera
                    "resolution": [1920, 1080],
                }
            }
        }

        config_file = config_dir / "misconfigured.json"
        with open(config_file, "w") as f:
            json.dump(misconfigured, f)

        # Load configuration through ConfigurationModule
        config_module = ConfigurationModule(
            config_dir=config_dir, enable_hot_reload=False
        )
        config_module.load_config(config_file)

        # Check that migration fixed the misconfiguration
        assert (
            config_module.get("vision.camera.video_source_type") == "camera"
        ), "Migration should set video_source_type to 'camera'"

        # loop_video should be removed
        assert (
            config_module.get("vision.camera.loop_video") is None
        ), "Migration should remove loop_video from camera config"

    def test_multiple_migrations_applied_sequentially(self, tmp_path):
        """Test that multiple migrations can be applied in sequence."""
        # This test ensures the migration system is extensible
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config = {
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                }
            }
        }

        config_file = config_dir / "test.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        # Load configuration through ConfigurationModule
        config_module = ConfigurationModule(
            config_dir=config_dir, enable_hot_reload=False
        )

        # Should apply all registered migrations
        result = config_module.load_config(config_file)
        assert result is True, "All migrations should be applied successfully"

        # Verify the migration was applied
        assert (
            config_module.get("vision.camera.video_source_type") == "file"
        ), "Migration should be applied"

    def test_migration_preserves_non_vision_config(self, tmp_path):
        """Test that migration preserves configuration outside vision module."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config = {
            "metadata": {"version": "1.0.0"},
            "system": {
                "debug": True,
                "environment": "development",
            },
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                },
                "detection": {
                    "min_ball_radius": 15,
                    "max_ball_radius": 45,
                },
            },
            "api": {
                "network": {
                    "host": "0.0.0.0",
                    "port": 9000,
                }
            },
        }

        config_file = config_dir / "full.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        # Load configuration through ConfigurationModule
        config_module = ConfigurationModule(
            config_dir=config_dir, enable_hot_reload=False
        )
        config_module.load_config(config_file)

        # Verify migration was applied
        assert (
            config_module.get("vision.camera.video_source_type") == "file"
        ), "Migration should be applied"

        # Verify other sections are preserved
        assert (
            config_module.get("metadata.version") == "1.0.0"
        ), "Metadata should be preserved"
        assert (
            config_module.get("system.debug") is True
        ), "System config should be preserved"
        assert (
            config_module.get("vision.detection.min_ball_radius") == 15
        ), "Detection config should be preserved"
        assert (
            config_module.get("api.network.port") == 9000
        ), "API config should be preserved"
