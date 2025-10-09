"""Tests for configuration migration system."""

import pytest

from backend.config.migrations.video_source_migration import (
    VideoSourceConfigMigration,
    migrate_config,
)


class TestVideoSourceConfigMigration:
    """Test suite for VideoSourceConfigMigration."""

    def test_should_migrate_string_device_id(self):
        """Test detection of legacy string device_id (file path)."""
        config = {
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                    "resolution": [1920, 1080],
                }
            }
        }

        migration = VideoSourceConfigMigration()
        assert migration.should_migrate(config) is True

    def test_should_migrate_int_device_id_without_type(self):
        """Test detection of int device_id without video_source_type."""
        config = {
            "vision": {
                "camera": {
                    "device_id": 0,
                    "resolution": [1920, 1080],
                }
            }
        }

        migration = VideoSourceConfigMigration()
        assert migration.should_migrate(config) is True

    def test_should_migrate_int_device_id_with_loop_video(self):
        """Test detection of int device_id with loop_video (misconfiguration)."""
        config = {
            "vision": {
                "camera": {
                    "device_id": 0,
                    "loop_video": True,
                    "resolution": [1920, 1080],
                }
            }
        }

        migration = VideoSourceConfigMigration()
        assert migration.should_migrate(config) is True

    def test_should_not_migrate_camera_config(self):
        """Test that properly configured camera source doesn't need migration."""
        config = {
            "vision": {
                "camera": {
                    "video_source_type": "camera",
                    "device_id": 0,
                    "resolution": [1920, 1080],
                }
            }
        }

        migration = VideoSourceConfigMigration()
        assert migration.should_migrate(config) is False

    def test_should_not_migrate_file_config(self):
        """Test that properly configured file source doesn't need migration."""
        config = {
            "vision": {
                "camera": {
                    "video_source_type": "file",
                    "video_file_path": "/path/to/video.mp4",
                    "loop_video": True,
                    "resolution": [1920, 1080],
                }
            }
        }

        migration = VideoSourceConfigMigration()
        assert migration.should_migrate(config) is False

    def test_should_not_migrate_empty_config(self):
        """Test that empty configuration doesn't need migration."""
        config = {}

        migration = VideoSourceConfigMigration()
        assert migration.should_migrate(config) is False

    def test_migrate_string_device_id_to_file(self):
        """Test migration of string device_id to file configuration."""
        config = {
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                    "resolution": [1920, 1080],
                    "fps": 30,
                }
            }
        }

        migration = VideoSourceConfigMigration()
        migrated = migration.migrate(config)

        # Check that video_source_type is set to file
        assert migrated["vision"]["camera"]["video_source_type"] == "file"

        # Check that video_file_path is set
        assert migrated["vision"]["camera"]["video_file_path"] == "/path/to/video.mp4"

        # Check that device_id was removed
        assert "device_id" not in migrated["vision"]["camera"]

        # Check that loop_video has a default
        assert "loop_video" in migrated["vision"]["camera"]

        # Check that other settings are preserved
        assert migrated["vision"]["camera"]["resolution"] == [1920, 1080]
        assert migrated["vision"]["camera"]["fps"] == 30

    def test_migrate_int_device_id_to_camera(self):
        """Test migration of int device_id to camera configuration."""
        config = {
            "vision": {
                "camera": {
                    "device_id": 1,
                    "resolution": [1920, 1080],
                    "fps": 30,
                }
            }
        }

        migration = VideoSourceConfigMigration()
        migrated = migration.migrate(config)

        # Check that video_source_type is set to camera
        assert migrated["vision"]["camera"]["video_source_type"] == "camera"

        # Check that device_id is preserved
        assert migrated["vision"]["camera"]["device_id"] == 1

        # Check that loop_video is NOT added for camera
        assert "loop_video" not in migrated["vision"]["camera"]

        # Check that other settings are preserved
        assert migrated["vision"]["camera"]["resolution"] == [1920, 1080]
        assert migrated["vision"]["camera"]["fps"] == 30

    def test_migrate_removes_loop_video_from_camera(self):
        """Test that loop_video is removed from camera configuration."""
        config = {
            "vision": {
                "camera": {
                    "device_id": 0,
                    "loop_video": True,  # This is invalid for camera
                    "resolution": [1920, 1080],
                }
            }
        }

        migration = VideoSourceConfigMigration()
        migrated = migration.migrate(config)

        # Check that video_source_type is set to camera
        assert migrated["vision"]["camera"]["video_source_type"] == "camera"

        # Check that loop_video was removed
        assert "loop_video" not in migrated["vision"]["camera"]

    def test_migrate_preserves_loop_video_for_file(self):
        """Test that loop_video is preserved for file configuration."""
        config = {
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                    "loop_video": True,
                    "resolution": [1920, 1080],
                }
            }
        }

        migration = VideoSourceConfigMigration()
        migrated = migration.migrate(config)

        # Check that loop_video is preserved
        assert migrated["vision"]["camera"]["loop_video"] is True

    def test_validate_deprecated_string_device_id(self):
        """Test validation warnings for deprecated string device_id."""
        config = {
            "vision": {
                "camera": {
                    "video_source_type": "file",
                    "device_id": "/path/to/video.mp4",  # Deprecated
                    "video_file_path": "/path/to/video.mp4",
                }
            }
        }

        migration = VideoSourceConfigMigration()
        warnings = migration.validate(config)

        # Should warn about deprecated device_id
        assert any("DEPRECATED" in w and "device_id" in w for w in warnings)

    def test_validate_missing_video_file_path(self):
        """Test validation warnings for missing video_file_path."""
        config = {
            "vision": {
                "camera": {
                    "video_source_type": "file",
                    # Missing video_file_path
                }
            }
        }

        migration = VideoSourceConfigMigration()
        warnings = migration.validate(config)

        # Should warn about missing video_file_path
        assert any("video_file_path" in w for w in warnings)

    def test_validate_loop_video_on_camera(self):
        """Test validation warnings for loop_video on camera source."""
        config = {
            "vision": {
                "camera": {
                    "video_source_type": "camera",
                    "device_id": 0,
                    "loop_video": True,  # Invalid for camera
                }
            }
        }

        migration = VideoSourceConfigMigration()
        warnings = migration.validate(config)

        # Should warn about loop_video on camera
        assert any("loop_video" in w and "camera" in w for w in warnings)

    def test_validate_valid_camera_config(self):
        """Test validation of valid camera configuration."""
        config = {
            "vision": {
                "camera": {
                    "video_source_type": "camera",
                    "device_id": 0,
                    "resolution": [1920, 1080],
                }
            }
        }

        migration = VideoSourceConfigMigration()
        warnings = migration.validate(config)

        # Should have no warnings
        assert len(warnings) == 0

    def test_validate_valid_file_config(self):
        """Test validation of valid file configuration."""
        config = {
            "vision": {
                "camera": {
                    "video_source_type": "file",
                    "video_file_path": "/path/to/video.mp4",
                    "loop_video": True,
                    "resolution": [1920, 1080],
                }
            }
        }

        migration = VideoSourceConfigMigration()
        warnings = migration.validate(config)

        # Should have no warnings
        assert len(warnings) == 0

    def test_get_suggested_configuration_camera(self):
        """Test suggested configuration generation for camera source."""
        config = {
            "vision": {
                "camera": {
                    "device_id": 0,
                }
            }
        }

        migration = VideoSourceConfigMigration()
        suggestion = migration.get_suggested_configuration(config)

        # Should contain camera configuration suggestion
        assert "camera" in suggestion.lower()
        assert "video_source_type" in suggestion
        assert "device_id" in suggestion

    def test_get_suggested_configuration_file(self):
        """Test suggested configuration generation for file source."""
        config = {
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                }
            }
        }

        migration = VideoSourceConfigMigration()
        suggestion = migration.get_suggested_configuration(config)

        # Should contain file configuration suggestion
        assert "file" in suggestion.lower()
        assert "video_source_type" in suggestion
        assert "video_file_path" in suggestion
        assert "loop_video" in suggestion

    def test_migrate_config_helper_function(self):
        """Test the convenience migrate_config function."""
        config = {
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                    "loop_video": True,
                }
            }
        }

        migrated, warnings = migrate_config(config)

        # Check migration was applied
        assert migrated["vision"]["camera"]["video_source_type"] == "file"
        assert migrated["vision"]["camera"]["video_file_path"] == "/path/to/video.mp4"

        # Warnings should be a list
        assert isinstance(warnings, list)

    def test_migrate_preserves_other_config_sections(self):
        """Test that migration preserves configuration outside vision.camera."""
        config = {
            "metadata": {"version": "1.0.0"},
            "system": {"debug": False},
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                },
                "detection": {
                    "min_ball_radius": 10,
                },
            },
            "api": {"port": 8000},
        }

        migration = VideoSourceConfigMigration()
        migrated = migration.migrate(config)

        # Check other sections are preserved
        assert migrated["metadata"]["version"] == "1.0.0"
        assert migrated["system"]["debug"] is False
        assert migrated["vision"]["detection"]["min_ball_radius"] == 10
        assert migrated["api"]["port"] == 8000

    def test_migrate_idempotent(self):
        """Test that running migration twice doesn't cause issues."""
        config = {
            "vision": {
                "camera": {
                    "device_id": "/path/to/video.mp4",
                    "loop_video": True,
                }
            }
        }

        migration = VideoSourceConfigMigration()

        # Apply migration first time
        migrated1 = migration.migrate(config.copy())

        # Apply migration second time (should be no-op)
        migrated2 = migration.migrate(migrated1.copy())

        # Results should be identical
        assert migrated1 == migrated2

    def test_validate_missing_video_source_type(self):
        """Test validation warns about missing video_source_type."""
        config = {
            "vision": {
                "camera": {
                    "device_id": 0,
                    # Missing video_source_type
                }
            }
        }

        migration = VideoSourceConfigMigration()
        warnings = migration.validate(config)

        # Should warn about missing video_source_type
        assert any("video_source_type" in w and "DEPRECATED" in w for w in warnings)

    def test_version_and_description(self):
        """Test that migration has version and description."""
        migration = VideoSourceConfigMigration()

        assert hasattr(migration, "version")
        assert hasattr(migration, "description")
        assert isinstance(migration.version, str)
        assert isinstance(migration.description, str)
        assert len(migration.version) > 0
        assert len(migration.description) > 0
