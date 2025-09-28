"""Calibration data persistence and management.

This module handles saving and loading calibration data, managing calibration
profiles, and providing backup/restore functionality for calibration settings.
"""

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CalibrationProfile:
    """A named calibration profile."""

    name: str
    description: str
    created: float
    modified: float
    keystone_data: dict[str, Any]
    geometric_data: dict[str, Any]
    metadata: dict[str, Any]


class CalibrationPersistence:
    """Handles persistence of calibration data and profiles.

    This class provides methods to:
    - Save and load calibration data to/from files
    - Manage multiple calibration profiles
    - Create backups and restore from backups
    - Validate calibration data integrity
    """

    def __init__(self, calibration_dir: Path = Path("config/calibration")):
        """Initialize calibration persistence.

        Args:
            calibration_dir: Directory to store calibration files
        """
        self.calibration_dir = Path(calibration_dir)
        self.current_profile_file = self.calibration_dir / "current.json"
        self.profiles_dir = self.calibration_dir / "profiles"
        self.backups_dir = self.calibration_dir / "backups"

        # Create directories if they don't exist
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(exist_ok=True)
        self.backups_dir.mkdir(exist_ok=True)

        # Current calibration state
        self.current_profile: Optional[CalibrationProfile] = None
        self.auto_backup: bool = True
        self.backup_count: int = 10

        logger.info(f"CalibrationPersistence initialized at {self.calibration_dir}")

    def save_current_calibration(
        self,
        keystone_data: dict[str, Any],
        geometric_data: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Save current calibration data.

        Args:
            keystone_data: Keystone calibration data
            geometric_data: Geometric calibration data
            metadata: Optional metadata

        Returns:
            True if saved successfully
        """
        try:
            # Create backup if auto-backup is enabled
            if self.auto_backup and self.current_profile_file.exists():
                self._create_backup()

            # Prepare calibration data
            calibration_data = {
                "version": "1.0.0",
                "timestamp": time.time(),
                "keystone": keystone_data,
                "geometric": geometric_data,
                "metadata": metadata or {},
            }

            # Write to file
            with open(self.current_profile_file, "w") as f:
                json.dump(calibration_data, f, indent=2)

            logger.info("Current calibration saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save current calibration: {e}")
            return False

    def load_current_calibration(
        self,
    ) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
        """Load current calibration data.

        Returns:
            Tuple of (keystone_data, geometric_data, metadata) or (None, None, None) if failed
        """
        try:
            if not self.current_profile_file.exists():
                logger.info("No current calibration file found")
                return None, None, None

            with open(self.current_profile_file) as f:
                data = json.load(f)

            # Validate data structure
            if not self._validate_calibration_data(data):
                logger.error("Invalid calibration data format")
                return None, None, None

            keystone_data = data.get("keystone", {})
            geometric_data = data.get("geometric", {})
            metadata = data.get("metadata", {})

            logger.info("Current calibration loaded successfully")
            return keystone_data, geometric_data, metadata

        except Exception as e:
            logger.error(f"Failed to load current calibration: {e}")
            return None, None, None

    def save_profile(
        self,
        name: str,
        description: str,
        keystone_data: dict[str, Any],
        geometric_data: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Save calibration as a named profile.

        Args:
            name: Profile name
            description: Profile description
            keystone_data: Keystone calibration data
            geometric_data: Geometric calibration data
            metadata: Optional metadata

        Returns:
            True if saved successfully
        """
        try:
            # Validate profile name
            if not self._validate_profile_name(name):
                logger.error(f"Invalid profile name: {name}")
                return False

            # Create profile
            profile = CalibrationProfile(
                name=name,
                description=description,
                created=time.time(),
                modified=time.time(),
                keystone_data=keystone_data.copy(),
                geometric_data=geometric_data.copy(),
                metadata=metadata.copy() if metadata else {},
            )

            # Save profile to file
            profile_file = self.profiles_dir / f"{name}.json"
            profile_data = asdict(profile)

            with open(profile_file, "w") as f:
                json.dump(profile_data, f, indent=2)

            logger.info(f"Profile '{name}' saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save profile '{name}': {e}")
            return False

    def load_profile(self, name: str) -> Optional[CalibrationProfile]:
        """Load a calibration profile by name.

        Args:
            name: Profile name

        Returns:
            CalibrationProfile object or None if failed
        """
        try:
            profile_file = self.profiles_dir / f"{name}.json"
            if not profile_file.exists():
                logger.error(f"Profile '{name}' not found")
                return None

            with open(profile_file) as f:
                data = json.load(f)

            # Create profile object
            profile = CalibrationProfile(**data)
            self.current_profile = profile

            logger.info(f"Profile '{name}' loaded successfully")
            return profile

        except Exception as e:
            logger.error(f"Failed to load profile '{name}': {e}")
            return None

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all available calibration profiles.

        Returns:
            List of profile information dictionaries
        """
        profiles = []

        try:
            for profile_file in self.profiles_dir.glob("*.json"):
                try:
                    with open(profile_file) as f:
                        data = json.load(f)

                    # Extract profile info
                    profile_info = {
                        "name": data.get("name", profile_file.stem),
                        "description": data.get("description", ""),
                        "created": data.get("created", 0),
                        "modified": data.get("modified", 0),
                        "file": str(profile_file),
                    }
                    profiles.append(profile_info)

                except Exception as e:
                    logger.warning(f"Failed to read profile {profile_file}: {e}")

            # Sort by modification time (newest first)
            profiles.sort(key=lambda x: x["modified"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list profiles: {e}")

        return profiles

    def delete_profile(self, name: str) -> bool:
        """Delete a calibration profile.

        Args:
            name: Profile name

        Returns:
            True if deleted successfully
        """
        try:
            profile_file = self.profiles_dir / f"{name}.json"
            if not profile_file.exists():
                logger.error(f"Profile '{name}' not found")
                return False

            # Create backup before deletion
            backup_file = self.backups_dir / f"deleted_{name}_{int(time.time())}.json"
            shutil.copy2(profile_file, backup_file)

            # Delete profile
            profile_file.unlink()

            logger.info(f"Profile '{name}' deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete profile '{name}': {e}")
            return False

    def export_profile(self, name: str, export_path: Path) -> bool:
        """Export a profile to an external file.

        Args:
            name: Profile name
            export_path: Path to export file

        Returns:
            True if exported successfully
        """
        try:
            profile_file = self.profiles_dir / f"{name}.json"
            if not profile_file.exists():
                logger.error(f"Profile '{name}' not found")
                return False

            # Copy profile file
            shutil.copy2(profile_file, export_path)

            logger.info(f"Profile '{name}' exported to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export profile '{name}': {e}")
            return False

    def import_profile(self, import_path: Path, new_name: Optional[str] = None) -> bool:
        """Import a profile from an external file.

        Args:
            import_path: Path to import file
            new_name: Optional new name for the profile

        Returns:
            True if imported successfully
        """
        try:
            if not import_path.exists():
                logger.error(f"Import file not found: {import_path}")
                return False

            # Load and validate profile data
            with open(import_path) as f:
                data = json.load(f)

            if not self._validate_profile_data(data):
                logger.error("Invalid profile data format")
                return False

            # Determine profile name
            profile_name = new_name or data.get("name", import_path.stem)

            # Update profile metadata
            data["name"] = profile_name
            data["modified"] = time.time()

            # Save to profiles directory
            profile_file = self.profiles_dir / f"{profile_name}.json"
            with open(profile_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Profile imported as '{profile_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to import profile: {e}")
            return False

    def _create_backup(self) -> bool:
        """Create a backup of the current calibration."""
        try:
            if not self.current_profile_file.exists():
                return True  # Nothing to backup

            # Create backup filename with timestamp
            timestamp = int(time.time())
            backup_file = self.backups_dir / f"calibration_backup_{timestamp}.json"

            # Copy current calibration to backup
            shutil.copy2(self.current_profile_file, backup_file)

            # Cleanup old backups
            self._cleanup_old_backups()

            logger.debug(f"Calibration backup created: {backup_file}")
            return True

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            return False

    def _cleanup_old_backups(self) -> None:
        """Remove old backup files beyond the backup count limit."""
        try:
            # Get all backup files
            backup_files = list(self.backups_dir.glob("calibration_backup_*.json"))

            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove excess backups
            for backup_file in backup_files[self.backup_count :]:
                backup_file.unlink()
                logger.debug(f"Removed old backup: {backup_file}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    def restore_from_backup(self, backup_file: Optional[Path] = None) -> bool:
        """Restore calibration from a backup file.

        Args:
            backup_file: Specific backup file to restore from (latest if None)

        Returns:
            True if restored successfully
        """
        try:
            if backup_file is None:
                # Find latest backup
                backup_files = list(self.backups_dir.glob("calibration_backup_*.json"))
                if not backup_files:
                    logger.error("No backup files found")
                    return False

                backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                backup_file = backup_files[0]

            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False

            # Validate backup data
            with open(backup_file) as f:
                data = json.load(f)

            if not self._validate_calibration_data(data):
                logger.error("Invalid backup data format")
                return False

            # Create backup of current state before restoring
            if self.current_profile_file.exists():
                emergency_backup = (
                    self.backups_dir / f"emergency_backup_{int(time.time())}.json"
                )
                shutil.copy2(self.current_profile_file, emergency_backup)

            # Restore from backup
            shutil.copy2(backup_file, self.current_profile_file)

            logger.info(f"Calibration restored from backup: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

    def _validate_calibration_data(self, data: dict[str, Any]) -> bool:
        """Validate calibration data structure.

        Args:
            data: Calibration data to validate

        Returns:
            True if valid
        """
        required_fields = ["version", "timestamp", "keystone", "geometric"]

        # Check required fields
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False

        # Check data types
        if not isinstance(data["keystone"], dict):
            logger.error("Keystone data must be a dictionary")
            return False

        if not isinstance(data["geometric"], dict):
            logger.error("Geometric data must be a dictionary")
            return False

        if not isinstance(data["timestamp"], (int, float)):
            logger.error("Timestamp must be a number")
            return False

        return True

    def _validate_profile_data(self, data: dict[str, Any]) -> bool:
        """Validate profile data structure.

        Args:
            data: Profile data to validate

        Returns:
            True if valid
        """
        required_fields = ["name", "keystone_data", "geometric_data"]

        # Check required fields
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required profile field: {field}")
                return False

        # Check data types
        if not isinstance(data["keystone_data"], dict):
            logger.error("Keystone data must be a dictionary")
            return False

        if not isinstance(data["geometric_data"], dict):
            logger.error("Geometric data must be a dictionary")
            return False

        return True

    def _validate_profile_name(self, name: str) -> bool:
        """Validate profile name.

        Args:
            name: Profile name to validate

        Returns:
            True if valid
        """
        if not name or not name.strip():
            return False

        # Check for invalid characters
        invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        if any(char in name for char in invalid_chars):
            return False

        # Check length
        return not len(name) > 100

    def get_backup_list(self) -> list[dict[str, Any]]:
        """Get list of available backup files.

        Returns:
            List of backup file information
        """
        backups = []

        try:
            for backup_file in self.backups_dir.glob("calibration_backup_*.json"):
                try:
                    stat = backup_file.stat()
                    backup_info = {
                        "file": str(backup_file),
                        "timestamp": stat.st_mtime,
                        "size": stat.st_size,
                        "name": backup_file.name,
                    }
                    backups.append(backup_info)

                except Exception as e:
                    logger.warning(f"Failed to get backup info for {backup_file}: {e}")

            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list backups: {e}")

        return backups

    def set_auto_backup(self, enabled: bool) -> None:
        """Enable or disable automatic backup creation.

        Args:
            enabled: Whether to enable auto-backup
        """
        self.auto_backup = enabled
        logger.info(f"Auto-backup {'enabled' if enabled else 'disabled'}")

    def set_backup_count(self, count: int) -> None:
        """Set the number of backup files to keep.

        Args:
            count: Number of backups to keep
        """
        if count < 1:
            raise ValueError("Backup count must be at least 1")

        self.backup_count = count
        self._cleanup_old_backups()
        logger.info(f"Backup count set to {count}")

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about calibration storage.

        Returns:
            Dictionary containing storage information
        """
        try:
            # Count files and calculate sizes
            current_size = (
                self.current_profile_file.stat().st_size
                if self.current_profile_file.exists()
                else 0
            )

            profile_files = list(self.profiles_dir.glob("*.json"))
            profiles_size = sum(f.stat().st_size for f in profile_files)

            backup_files = list(self.backups_dir.glob("*.json"))
            backups_size = sum(f.stat().st_size for f in backup_files)

            return {
                "calibration_dir": str(self.calibration_dir),
                "current_calibration_size": current_size,
                "profiles_count": len(profile_files),
                "profiles_size": profiles_size,
                "backups_count": len(backup_files),
                "backups_size": backups_size,
                "total_size": current_size + profiles_size + backups_size,
                "auto_backup_enabled": self.auto_backup,
                "backup_count_limit": self.backup_count,
            }

        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {}
