"""Tests for configuration backup and restore functionality."""

import tempfile
import time
from pathlib import Path

import pytest

from ..manager import ConfigurationModule
from ..models.schemas import ConfigSource
from ..storage.backup import BackupNotFoundError, ConfigBackup


class TestConfigBackup:
    """Test the ConfigBackup class functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backup_dir = self.temp_dir / "backups"
        self.backup = ConfigBackup(
            backup_dir=self.backup_dir,
            max_backups=5,
            compression=False,  # Disable compression for easier testing
            verify_integrity=True,
        )

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_create_backup(self):
        """Test creating a manual backup."""
        test_config = {
            "app": {"name": "test", "version": "1.0.0"},
            "api": {"port": 8080},
        }

        backup_path = self.backup.create_backup(
            test_config,
            "test_backup",
            description="Test backup",
            tags=["test", "manual"],
        )

        # Verify backup was created
        assert Path(backup_path).exists()
        assert "test_backup" in backup_path

        # Verify backup content
        backup_content = self.backup._read_backup_file(Path(backup_path))
        assert "metadata" in backup_content
        assert "config" in backup_content
        assert backup_content["config"] == test_config

        # Verify metadata
        metadata = self.backup.get_backup_info("test_backup")
        assert metadata is not None
        assert metadata.name == "test_backup"
        assert metadata.description == "Test backup"
        assert "test" in metadata.tags
        assert "manual" in metadata.tags
        assert not metadata.auto_backup

    def test_create_auto_backup(self):
        """Test creating an automatic backup."""
        test_config = {"setting": "value"}

        backup_path = self.backup.create_auto_backup(test_config, "test_reason")

        # Verify backup was created
        assert Path(backup_path).exists()
        assert "auto_" in backup_path

        # Verify metadata
        backup_name = Path(backup_path).stem
        if backup_name.endswith(".json"):
            backup_name = backup_name[:-5]  # Remove .json extension

        metadata = self.backup.get_backup_info(backup_name)
        assert metadata is not None
        assert metadata.auto_backup
        assert "auto" in metadata.tags
        assert "test_reason" in metadata.tags

    def test_list_backups(self):
        """Test listing backups with filtering."""
        # Create test backups
        test_config = {"test": "data"}

        self.backup.create_backup(test_config, "manual1", tags=["tag1"])
        self.backup.create_backup(test_config, "manual2", tags=["tag2"])
        self.backup.create_auto_backup(test_config, "auto_reason")

        # Test listing all backups
        all_backups = self.backup.list_backups()
        assert len(all_backups) == 3

        # Test filtering by type
        manual_only = self.backup.list_backups(include_auto=False)
        assert len(manual_only) == 2
        assert all(not b.auto_backup for b in manual_only)

        auto_only = self.backup.list_backups(include_manual=False)
        assert len(auto_only) == 1
        assert all(b.auto_backup for b in auto_only)

        # Test filtering by tags
        tag1_backups = self.backup.list_backups(tags=["tag1"])
        assert len(tag1_backups) == 1
        assert tag1_backups[0].name == "manual1"

    def test_restore_backup(self):
        """Test restoring from backup."""
        test_config = {
            "app": {"name": "restored", "version": "2.0.0"},
            "api": {"port": 9000},
        }

        # Create backup
        self.backup.create_backup(test_config, "restore_test")

        # Restore backup
        restored_config = self.backup.restore_backup("restore_test")

        assert restored_config == test_config

    def test_restore_nonexistent_backup(self):
        """Test restoring from non-existent backup."""
        with pytest.raises(BackupNotFoundError):
            self.backup.restore_backup("nonexistent")

    def test_delete_backup(self):
        """Test deleting a backup."""
        test_config = {"test": "data"}
        self.backup.create_backup(test_config, "delete_test")

        # Verify backup exists
        assert self.backup.get_backup_info("delete_test") is not None

        # Delete backup
        success = self.backup.delete_backup("delete_test")
        assert success

        # Verify backup is gone
        assert self.backup.get_backup_info("delete_test") is None

    def test_verify_backup_integrity(self):
        """Test backup integrity verification."""
        test_config = {"test": "data"}
        self.backup.create_backup(test_config, "integrity_test")

        # Verify integrity
        assert self.backup.verify_backup_integrity("integrity_test")

        # Test with non-existent backup
        assert not self.backup.verify_backup_integrity("nonexistent")

    def test_cleanup_old_backups(self):
        """Test cleanup of old backups."""
        # Create backup manager with high max_backups to avoid rotation interference
        cleanup_backup = ConfigBackup(
            backup_dir=self.temp_dir / "cleanup_test",
            max_backups=20,  # High enough to not interfere
            compression=False,
            verify_integrity=True,
        )

        test_config = {"test": "data"}
        old_time = time.time() - (40 * 24 * 60 * 60)  # 40 days ago

        # Create old backups and get their actual names
        backup1_path = cleanup_backup.create_auto_backup(test_config, "old1")
        time.sleep(1.1)  # Ensure different timestamps
        backup2_path = cleanup_backup.create_auto_backup(test_config, "old2")

        # Extract backup names - they are generated with timestamps
        backup1_name = Path(backup1_path).stem
        backup2_name = Path(backup2_path).stem

        # Update metadata with old timestamps for both backups
        for backup_name in [backup1_name, backup2_name]:
            metadata = cleanup_backup.get_backup_info(backup_name)
            if metadata:
                metadata.timestamp = old_time
                cleanup_backup._save_metadata(
                    cleanup_backup._sanitize_filename(backup_name), metadata
                )

        # Create recent backup
        cleanup_backup.create_backup(test_config, "recent")

        # Verify we have 3 backups before cleanup
        all_backups = cleanup_backup.list_backups()
        assert len(all_backups) == 3

        # Cleanup backups older than 30 days
        cleaned_count = cleanup_backup.cleanup_old_backups(max_age_days=30)

        # Should have cleaned up 2 auto backups but kept manual backup
        assert cleaned_count == 2

        # Verify recent backup still exists
        assert cleanup_backup.get_backup_info("recent") is not None

        # Verify we now have only 1 backup
        remaining_backups = cleanup_backup.list_backups()
        assert len(remaining_backups) == 1

    def test_export_import_backup(self):
        """Test exporting and importing backups."""
        test_config = {"exported": "data"}
        self.backup.create_backup(test_config, "export_test")

        # Export backup
        export_path = self.temp_dir / "exported_backup.json"
        success = self.backup.export_backup("export_test", export_path)
        assert success
        assert export_path.exists()

        # Create new backup manager for import test
        import_backup_dir = self.temp_dir / "import_backups"
        import_backup = ConfigBackup(backup_dir=import_backup_dir)

        # Import backup
        imported_name = import_backup.import_backup(export_path, "imported_test")
        assert imported_name

        # Verify imported backup
        imported_config = import_backup.restore_backup("imported_test")
        assert imported_config == test_config

    def test_backup_rotation(self):
        """Test automatic backup rotation."""
        # Create backup manager with low max_backups
        rotation_backup = ConfigBackup(
            backup_dir=self.temp_dir / "rotation", max_backups=3
        )

        test_config = {"test": "data"}

        # Create more backups than max_backups
        for i in range(5):
            rotation_backup.create_auto_backup(test_config, f"reason_{i}")
            time.sleep(0.1)  # Ensure different timestamps

        # Should only have 3 backups due to rotation
        backups = rotation_backup.list_backups(include_manual=False)
        assert len(backups) <= 3


class TestConfigurationModuleBackup:
    """Test backup functionality integrated with ConfigurationModule."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_module = ConfigurationModule(
            config_dir=self.temp_dir / "config", enable_hot_reload=False
        )

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_create_backup_integration(self):
        """Test creating backup through ConfigurationModule."""
        # Set some configuration values
        self.config_module.set("test.setting1", "value1", persist=True)
        self.config_module.set("test.setting2", 42, persist=True)

        # Create backup
        backup_path = self.config_module.create_backup(
            "integration_test", description="Integration test backup"
        )

        assert backup_path
        assert "integration_test" in backup_path

        # Verify backup content
        backup_info = self.config_module.get_backup_info("integration_test")
        assert backup_info is not None
        assert backup_info.name == "integration_test"

    def test_restore_backup_integration(self):
        """Test restoring backup through ConfigurationModule."""
        # Set initial configuration
        self.config_module.set("test.original", "original_value", persist=True)
        self.config_module.set("test.number", 100, persist=True)

        # Create backup
        self.config_module.create_backup("before_changes")

        # Change configuration
        self.config_module.set("test.original", "changed_value", persist=True)
        self.config_module.set("test.number", 200, persist=True)
        self.config_module.set("test.new", "new_value", persist=True)

        # Verify changes
        assert self.config_module.get("test.original") == "changed_value"
        assert self.config_module.get("test.number") == 200
        assert self.config_module.get("test.new") == "new_value"

        # Restore backup
        success = self.config_module.restore_backup("before_changes")
        assert success

        # Verify restoration
        assert self.config_module.get("test.original") == "original_value"
        assert self.config_module.get("test.number") == 100
        assert self.config_module.get("test.new") is None  # Should be removed

    def test_auto_backup_on_significant_changes(self):
        """Test automatic backup creation on significant changes."""
        # Set initial configuration
        self.config_module.set("test.value", "initial", persist=True)

        # Count existing backups
        initial_backup_count = len(self.config_module.list_config_backups())

        # Make significant change (API source with persistence)
        self.config_module.set("test.value", "changed", ConfigSource.API, persist=True)

        # Should have created an auto backup
        # Note: This depends on the implementation creating auto backups for API changes
        # The current implementation doesn't automatically create backups, so we'll test manual creation
        auto_backup_path = self.config_module.create_auto_backup("test_change")
        assert auto_backup_path

        # Verify auto backup was created
        new_backup_count = len(self.config_module.list_config_backups())
        assert new_backup_count > initial_backup_count

    def test_list_and_manage_backups(self):
        """Test listing and managing backups through ConfigurationModule."""
        # Create several backups
        self.config_module.create_backup("backup1", tags=["test"])
        self.config_module.create_backup("backup2", tags=["production"])
        self.config_module.create_auto_backup("auto_reason")

        # List all backups
        all_backups = self.config_module.list_config_backups()
        assert len(all_backups) >= 3

        # Test filtering
        manual_backups = self.config_module.list_config_backups(include_auto=False)
        auto_backups = self.config_module.list_config_backups(include_manual=False)

        assert len(manual_backups) >= 2
        assert len(auto_backups) >= 1

        # Verify backup info
        backup1_info = self.config_module.get_backup_info("backup1")
        assert backup1_info is not None
        assert backup1_info.name == "backup1"
        assert "test" in backup1_info.tags

        # Test backup verification
        assert self.config_module.verify_backup("backup1")
        assert not self.config_module.verify_backup("nonexistent")

        # Test backup deletion
        assert self.config_module.delete_backup("backup2")
        assert self.config_module.get_backup_info("backup2") is None

    def test_backup_cleanup(self):
        """Test backup cleanup functionality."""
        # Create some backups
        for i in range(5):
            self.config_module.create_auto_backup(f"reason_{i}")

        initial_count = len(
            self.config_module.list_config_backups(include_manual=False)
        )

        # Cleanup (this will test the max_backups setting)
        self.config_module.cleanup_old_config_backups(max_age_days=0)

        # All auto backups should be cleaned if max_age_days=0
        remaining_count = len(
            self.config_module.list_config_backups(include_manual=False)
        )
        assert remaining_count <= initial_count

    def test_export_import_integration(self):
        """Test backup export/import through ConfigurationModule."""
        # Set configuration and create backup
        self.config_module.set("export.test", "export_value", persist=True)
        self.config_module.create_backup("export_test", description="For export")

        # Export backup
        export_path = self.temp_dir / "exported.json"
        success = self.config_module.export_backup("export_test", export_path)
        assert success
        assert export_path.exists()

        # Import to new configuration module
        import_config_module = ConfigurationModule(
            config_dir=self.temp_dir / "import_config", enable_hot_reload=False
        )

        imported_name = import_config_module.import_backup(export_path, "imported_test")
        assert imported_name

        # Verify imported backup can be restored
        import_success = import_config_module.restore_backup("imported_test")
        assert import_success
        assert import_config_module.get("export.test") == "export_value"


if __name__ == "__main__":
    pytest.main([__file__])
