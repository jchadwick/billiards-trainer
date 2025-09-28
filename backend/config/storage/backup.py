"""Configuration backup and restore system.

Provides comprehensive backup management for configuration data including:
- Manual and automatic backups
- Backup rotation and cleanup
- Integrity verification
- Metadata tracking
- Safe restore operations

"""

import gzip
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class BackupMetadata:
    """Backup metadata information."""

    name: str
    timestamp: float
    description: Optional[str]
    version: str
    checksum: str
    size_bytes: int
    config_keys_count: int
    source: str
    auto_backup: bool = False
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    @property
    def creation_date(self) -> datetime:
        """Get creation date as datetime object."""
        return datetime.fromtimestamp(self.timestamp)

    @property
    def age_days(self) -> float:
        """Get backup age in days."""
        return (time.time() - self.timestamp) / 86400

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackupMetadata":
        """Create from dictionary."""
        return cls(**data)


class BackupError(Exception):
    """Base exception for backup operations."""

    pass


class BackupCorruptedError(BackupError):
    """Raised when backup integrity check fails."""

    pass


class BackupNotFoundError(BackupError):
    """Raised when backup is not found."""

    pass


class ConfigBackup:
    """Configuration backup and restore manager.

    Provides comprehensive backup functionality including:
    - Named and automatic backups
    - Backup rotation and cleanup
    - Integrity verification
    - Metadata tracking
    - Safe restore operations
    """

    def __init__(
        self,
        backup_dir: Path = None,
        max_backups: int = 10,
        compression: bool = True,
        verify_integrity: bool = True,
    ):
        """Initialize backup manager.

        Args:
            backup_dir: Directory to store backups (defaults to config/backups)
            max_backups: Maximum number of automatic backups to keep
            compression: Whether to compress backup files
            verify_integrity: Whether to verify backup integrity
        """
        self.backup_dir = backup_dir or Path("config/backups")
        self.max_backups = max_backups
        self.compression = compression
        self.verify_integrity = verify_integrity

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.backup_dir / "manual").mkdir(exist_ok=True)
        (self.backup_dir / "auto").mkdir(exist_ok=True)
        (self.backup_dir / "metadata").mkdir(exist_ok=True)

    def create_backup(
        self,
        config_data: dict[str, Any],
        backup_name: str,
        description: Optional[str] = None,
        tags: list[str] = None,
    ) -> str:
        """Create a named backup of configuration data.

        Args:
            config_data: Configuration data to backup
            backup_name: Name for the backup
            description: Optional description
            tags: Optional tags for organization

        Returns:
            Path to created backup file

        Raises:
            BackupError: If backup creation fails
        """
        try:
            # Sanitize backup name
            safe_name = self._sanitize_filename(backup_name)
            timestamp = time.time()

            # Create backup file path
            extension = ".json.gz" if self.compression else ".json"
            backup_filename = f"{safe_name}_{int(timestamp)}{extension}"
            backup_path = self.backup_dir / "manual" / backup_filename

            # Create backup content
            backup_content = {
                "metadata": {
                    "name": backup_name,
                    "timestamp": timestamp,
                    "description": description,
                    "version": "1.0.0",
                    "source": "manual",
                    "tags": tags or [],
                },
                "config": config_data,
            }

            # Write backup file
            self._write_backup_file(backup_path, backup_content)

            # Calculate metadata
            checksum = self._calculate_checksum(backup_path)
            size_bytes = backup_path.stat().st_size

            # Create and save metadata
            metadata = BackupMetadata(
                name=backup_name,
                timestamp=timestamp,
                description=description,
                version="1.0.0",
                checksum=checksum,
                size_bytes=size_bytes,
                config_keys_count=len(config_data),
                source="manual",
                auto_backup=False,
                tags=tags or [],
            )

            self._save_metadata(safe_name, metadata)

            return str(backup_path)

        except Exception as e:
            raise BackupError(f"Failed to create backup '{backup_name}': {e}")

    def create_auto_backup(
        self, config_data: dict[str, Any], reason: str = "automatic"
    ) -> str:
        """Create an automatic timestamped backup.

        Args:
            config_data: Configuration data to backup
            reason: Reason for automatic backup

        Returns:
            Path to created backup file
        """
        timestamp = time.time()
        dt = datetime.fromtimestamp(timestamp)

        # Create auto backup name
        auto_name = f"auto_{dt.strftime('%Y%m%d_%H%M%S')}"

        try:
            # Create backup file path
            extension = ".json.gz" if self.compression else ".json"
            backup_filename = f"{auto_name}{extension}"
            backup_path = self.backup_dir / "auto" / backup_filename

            # Create backup content
            backup_content = {
                "metadata": {
                    "name": auto_name,
                    "timestamp": timestamp,
                    "description": f"Automatic backup: {reason}",
                    "version": "1.0.0",
                    "source": "automatic",
                    "tags": ["auto", reason],
                },
                "config": config_data,
            }

            # Write backup file
            self._write_backup_file(backup_path, backup_content)

            # Calculate metadata
            checksum = self._calculate_checksum(backup_path)
            size_bytes = backup_path.stat().st_size

            # Create and save metadata
            metadata = BackupMetadata(
                name=auto_name,
                timestamp=timestamp,
                description=f"Automatic backup: {reason}",
                version="1.0.0",
                checksum=checksum,
                size_bytes=size_bytes,
                config_keys_count=len(config_data),
                source="automatic",
                auto_backup=True,
                tags=["auto", reason],
            )

            self._save_metadata(auto_name, metadata)

            # Clean up old automatic backups
            self._cleanup_auto_backups()

            return str(backup_path)

        except Exception as e:
            raise BackupError(f"Failed to create automatic backup: {e}")

    def list_backups(
        self,
        include_auto: bool = True,
        include_manual: bool = True,
        tags: list[str] = None,
    ) -> list[BackupMetadata]:
        """List available backups.

        Args:
            include_auto: Include automatic backups
            include_manual: Include manual backups
            tags: Filter by tags (all tags must match)

        Returns:
            List of backup metadata, sorted by timestamp (newest first)
        """
        backups = []

        # Scan metadata directory
        metadata_dir = self.backup_dir / "metadata"
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file) as f:
                        data = json.load(f)

                    metadata = BackupMetadata.from_dict(data)

                    # Filter by type
                    if metadata.auto_backup and not include_auto:
                        continue
                    if not metadata.auto_backup and not include_manual:
                        continue

                    # Filter by tags
                    if tags and not all(tag in metadata.tags for tag in tags):
                        continue

                    backups.append(metadata)

                except Exception:
                    # Skip corrupted metadata files
                    continue

        # Sort by timestamp (newest first)
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)

    def restore_backup(
        self, backup_name: str, validate_before_restore: bool = True
    ) -> dict[str, Any]:
        """Restore configuration from backup.

        Args:
            backup_name: Name of backup to restore
            validate_before_restore: Whether to validate backup integrity first

        Returns:
            Restored configuration data

        Raises:
            BackupNotFoundError: If backup is not found
            BackupCorruptedError: If backup integrity check fails
        """
        # Find backup file
        backup_path = self._find_backup_file(backup_name)
        if not backup_path:
            raise BackupNotFoundError(f"Backup '{backup_name}' not found")

        # Validate integrity if requested
        if validate_before_restore and self.verify_integrity:
            if not self._verify_backup_integrity(backup_name):
                raise BackupCorruptedError(f"Backup '{backup_name}' is corrupted")

        try:
            # Read backup file
            backup_content = self._read_backup_file(backup_path)

            # Extract configuration data
            if "config" in backup_content:
                return backup_content["config"]
            else:
                # Legacy format support
                return backup_content

        except Exception as e:
            raise BackupError(f"Failed to restore backup '{backup_name}': {e}")

    def delete_backup(self, backup_name: str) -> bool:
        """Delete a backup.

        Args:
            backup_name: Name of backup to delete

        Returns:
            True if backup was deleted, False if not found
        """
        try:
            # Find and delete backup file
            backup_path = self._find_backup_file(backup_name)
            if backup_path and backup_path.exists():
                backup_path.unlink()

            # Delete metadata file
            safe_name = self._sanitize_filename(backup_name)
            metadata_path = self.backup_dir / "metadata" / f"{safe_name}.json"
            if metadata_path.exists():
                metadata_path.unlink()

            return True

        except Exception:
            return False

    def verify_backup_integrity(self, backup_name: str) -> bool:
        """Verify backup integrity.

        Args:
            backup_name: Name of backup to verify

        Returns:
            True if backup is valid, False otherwise
        """
        return self._verify_backup_integrity(backup_name)

    def get_backup_info(self, backup_name: str) -> Optional[BackupMetadata]:
        """Get backup metadata.

        Args:
            backup_name: Name of backup

        Returns:
            Backup metadata or None if not found
        """
        safe_name = self._sanitize_filename(backup_name)
        metadata_path = self.backup_dir / "metadata" / f"{safe_name}.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                data = json.load(f)
            return BackupMetadata.from_dict(data)
        except Exception:
            return None

    def cleanup_old_backups(
        self, max_age_days: int = 30, keep_manual: bool = True
    ) -> int:
        """Clean up old backups.

        Args:
            max_age_days: Maximum age for automatic backups
            keep_manual: Whether to keep manual backups regardless of age

        Returns:
            Number of backups cleaned up
        """
        cleaned_count = 0
        cutoff_time = time.time() - (max_age_days * 86400)

        for backup in self.list_backups():
            # Skip manual backups if configured to keep them
            if not backup.auto_backup and keep_manual:
                continue

            # Delete old backups
            if backup.timestamp < cutoff_time and self.delete_backup(backup.name):
                cleaned_count += 1

        return cleaned_count

    def export_backup(self, backup_name: str, export_path: Path) -> bool:
        """Export backup to external location.

        Args:
            backup_name: Name of backup to export
            export_path: Path to export to

        Returns:
            True if export successful, False otherwise
        """
        try:
            backup_path = self._find_backup_file(backup_name)
            if not backup_path:
                return False

            # Copy backup file
            shutil.copy2(backup_path, export_path)
            return True

        except Exception:
            return False

    def import_backup(self, import_path: Path, backup_name: str = None) -> str:
        """Import backup from external location.

        Args:
            import_path: Path to backup file to import
            backup_name: Optional name for imported backup

        Returns:
            Name of imported backup

        Raises:
            BackupError: If import fails
        """
        try:
            if not import_path.exists():
                raise BackupError(f"Import file not found: {import_path}")

            # Read backup content to validate
            backup_content = self._read_backup_file(import_path)

            # Extract or generate backup name
            if backup_name is None:
                if (
                    "metadata" in backup_content
                    and "name" in backup_content["metadata"]
                ):
                    backup_name = backup_content["metadata"]["name"]
                else:
                    backup_name = f"imported_{int(time.time())}"

            # Extract config data
            config_data = backup_content.get("config", backup_content)

            # Create backup
            return self.create_backup(
                config_data,
                backup_name,
                description=f"Imported from {import_path.name}",
                tags=["imported"],
            )

        except Exception as e:
            raise BackupError(f"Failed to import backup: {e}")

    # Private helper methods

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            name = name.replace(char, "_")
        return name.strip("._ ")

    def _write_backup_file(self, path: Path, content: dict[str, Any]) -> None:
        """Write backup file with optional compression."""
        json_data = json.dumps(content, indent=2, default=str)

        if self.compression and path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(json_data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_data)

    def _read_backup_file(self, path: Path) -> dict[str, Any]:
        """Read backup file with optional decompression."""
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Try to detect if file is gzipped by checking magic bytes
            try:
                with open(path, "rb") as f:
                    magic = f.read(2)
                    if magic == b"\x1f\x8b":  # gzip magic number
                        f.seek(0)
                        with gzip.open(f, "rt", encoding="utf-8") as gz_f:
                            return json.load(gz_f)
                    else:
                        f.seek(0)
                        content = f.read().decode("utf-8")
                        return json.loads(content)
            except UnicodeDecodeError:
                # If it fails to decode as UTF-8, try as gzip
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return json.load(f)

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _save_metadata(self, safe_name: str, metadata: BackupMetadata) -> None:
        """Save backup metadata."""
        metadata_path = self.backup_dir / "metadata" / f"{safe_name}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

    def _find_backup_file(self, backup_name: str) -> Optional[Path]:
        """Find backup file by name."""
        safe_name = self._sanitize_filename(backup_name)

        # Search in manual backups
        manual_dir = self.backup_dir / "manual"
        if manual_dir.exists():
            for pattern in [f"{safe_name}_*.json", f"{safe_name}_*.json.gz"]:
                matches = list(manual_dir.glob(pattern))
                if matches:
                    return matches[0]  # Return first match

        # Search in auto backups
        auto_dir = self.backup_dir / "auto"
        if auto_dir.exists():
            for pattern in [f"{safe_name}.json", f"{safe_name}.json.gz"]:
                backup_path = auto_dir / pattern
                if backup_path.exists():
                    return backup_path

        return None

    def _verify_backup_integrity(self, backup_name: str) -> bool:
        """Verify backup file integrity."""
        try:
            # Get metadata
            metadata = self.get_backup_info(backup_name)
            if not metadata:
                return False

            # Find backup file
            backup_path = self._find_backup_file(backup_name)
            if not backup_path:
                return False

            # Verify checksum
            current_checksum = self._calculate_checksum(backup_path)
            if current_checksum != metadata.checksum:
                return False

            # Try to read and parse the backup
            backup_content = self._read_backup_file(backup_path)

            # Basic structure validation
            if "metadata" in backup_content and "config" in backup_content:
                return True
            elif isinstance(backup_content, dict):
                # Legacy format
                return True

            return False

        except Exception:
            return False

    def _cleanup_auto_backups(self) -> None:
        """Clean up old automatic backups beyond max_backups."""
        auto_backups = [
            b for b in self.list_backups(include_manual=False) if b.auto_backup
        ]

        if len(auto_backups) > self.max_backups:
            # Sort by timestamp (oldest first for deletion)
            auto_backups.sort(key=lambda x: x.timestamp)

            # Delete oldest backups
            for backup in auto_backups[: -self.max_backups]:
                self.delete_backup(backup.name)


# Convenience aliases for backward compatibility
BackupManager = ConfigBackup
