"""Configuration persistence.

Provides robust configuration persistence with support for multiple formats,
atomic writes, error handling, and profile management.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Union

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

import contextlib

from ..models.schemas import ApplicationConfig, ConfigFormat, ConfigProfileEnhanced
from .encryption import ConfigEncryption, ConfigEncryptionError


class ConfigPersistenceError(Exception):
    """Configuration persistence error."""

    pass


class ConfigPersistence:
    """Configuration persistence manager.

    Provides robust configuration saving and loading with:
    - Multiple format support (JSON, YAML)
    - Atomic writes for data safety
    - Comprehensive error handling
    - Profile management
    - Backup functionality
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        encryption: Optional[ConfigEncryption] = None,
        enable_encryption: bool = False,
    ):
        """Initialize persistence manager.

        Args:
            base_dir: Base directory for configuration files (defaults to 'config')
            logger: Logger instance (creates one if not provided)
            encryption: ConfigEncryption instance for secure storage
            enable_encryption: Whether to enable encryption for sensitive fields
        """
        self.base_dir = Path(base_dir) if base_dir else Path("config")
        self.profiles_dir = self.base_dir / "profiles"
        self.backups_dir = self.base_dir / "backups"

        self.logger = logger or logging.getLogger(__name__)
        self.encryption = encryption
        self.enable_encryption = enable_encryption

        # Initialize encryption if enabled
        if self.enable_encryption and not self.encryption:
            try:
                self.encryption = ConfigEncryption()
                self.encryption.initialize()
                self.logger.info("Configuration encryption initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize encryption: {e}")
                self.enable_encryption = False

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for directory in [self.base_dir, self.profiles_dir, self.backups_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def _detect_format(self, file_path: Path) -> ConfigFormat:
        """Detect configuration format from file extension.

        Args:
            file_path: Path to the configuration file

        Returns:
            Detected configuration format

        Raises:
            ConfigPersistenceError: If format cannot be detected
        """
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return ConfigFormat.JSON
        elif suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ConfigPersistenceError(
                    "YAML format requested but PyYAML is not installed. "
                    "Install with: pip install PyYAML"
                )
            return ConfigFormat.YAML
        else:
            # Default to JSON for unknown extensions
            self.logger.warning(
                f"Unknown file extension '{suffix}', defaulting to JSON format"
            )
            return ConfigFormat.JSON

    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """Create a backup of an existing file.

        Args:
            file_path: Path to the file to backup

        Returns:
            Path to the backup file, or None if backup failed/not needed
        """
        if not file_path.exists():
            return None

        try:
            timestamp = int(time.time())
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.backups_dir / backup_name

            # Copy the file to backup location
            import shutil

            shutil.copy2(file_path, backup_path)

            self.logger.info(f"Created backup: {backup_path}")
            return backup_path

        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            return None

    def _serialize_data(self, data: dict[str, Any], format: ConfigFormat) -> str:
        """Serialize data to the specified format.

        Args:
            data: Data to serialize
            format: Target format

        Returns:
            Serialized data as string

        Raises:
            ConfigPersistenceError: If serialization fails
        """
        try:
            if format == ConfigFormat.JSON:
                return json.dumps(data, indent=2, default=str, ensure_ascii=False)
            elif format == ConfigFormat.YAML:
                if not YAML_AVAILABLE:
                    raise ConfigPersistenceError("YAML support not available")
                return yaml.dump(
                    data, default_flow_style=False, allow_unicode=True, sort_keys=False
                )
            else:
                raise ConfigPersistenceError(f"Unsupported format: {format}")

        except Exception as e:
            raise ConfigPersistenceError(f"Failed to serialize data as {format}: {e}")

    def _deserialize_data(self, content: str, format: ConfigFormat) -> dict[str, Any]:
        """Deserialize data from the specified format.

        Args:
            content: Content to deserialize
            format: Source format

        Returns:
            Deserialized data

        Raises:
            ConfigPersistenceError: If deserialization fails
        """
        try:
            if format == ConfigFormat.JSON:
                return json.loads(content)
            elif format == ConfigFormat.YAML:
                if not YAML_AVAILABLE:
                    raise ConfigPersistenceError("YAML support not available")
                return yaml.safe_load(content) or {}
            else:
                raise ConfigPersistenceError(f"Unsupported format: {format}")

        except Exception as e:
            raise ConfigPersistenceError(f"Failed to deserialize data as {format}: {e}")

    def _atomic_write(self, file_path: Path, content: str) -> None:
        """Perform atomic write operation.

        Args:
            file_path: Target file path
            content: Content to write

        Raises:
            ConfigPersistenceError: If write operation fails
        """
        try:
            # Create temporary file in the same directory as target
            temp_dir = file_path.parent
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=temp_dir,
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Ensure data is written to disk
                temp_path = Path(temp_file.name)

            # Atomic rename operation
            temp_path.replace(file_path)
            self.logger.debug(f"Atomic write completed: {file_path}")

        except Exception as e:
            # Clean up temporary file if it exists
            if "temp_path" in locals() and temp_path.exists():
                with contextlib.suppress(Exception):
                    temp_path.unlink()

            raise ConfigPersistenceError(f"Atomic write failed for {file_path}: {e}")

    def save_config(
        self,
        config_data: Union[dict[str, Any], ApplicationConfig],
        file_path: Union[str, Path],
        format: Optional[ConfigFormat] = None,
        create_backup: bool = True,
    ) -> bool:
        """Save configuration to file.

        Args:
            config_data: Configuration data to save
            file_path: Path to save configuration
            format: File format (auto-detected if None)
            create_backup: Whether to create backup of existing file

        Returns:
            True if saved successfully, False otherwise
        """
        file_path = Path(file_path)

        try:
            # Auto-detect format if not specified
            if format is None:
                format = self._detect_format(file_path)

            # Convert ApplicationConfig to dict if needed
            if isinstance(config_data, ApplicationConfig):
                data = config_data.model_dump()
                # Update metadata
                data.setdefault("metadata", {})
                data["metadata"]["modified"] = time.time()
            else:
                data = dict(config_data)

            # Apply encryption if enabled
            if (
                self.enable_encryption
                and self.encryption
                and self.encryption.is_encryption_enabled()
            ):
                try:
                    data = self.encryption.encrypt_config_dict(data)
                    self.logger.debug(
                        "Applied encryption to sensitive configuration fields"
                    )
                except ConfigEncryptionError as e:
                    self.logger.warning(
                        f"Encryption failed, saving without encryption: {e}"
                    )

            # Create backup if requested and file exists
            if create_backup:
                self._create_backup(file_path)

            # Serialize data
            content = self._serialize_data(data, format)

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Perform atomic write
            self._atomic_write(file_path, content)

            self.logger.info(f"Configuration saved successfully: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save configuration to {file_path}: {e}")
            return False

    def load_config(
        self, file_path: Union[str, Path], format: Optional[ConfigFormat] = None
    ) -> dict[str, Any]:
        """Load configuration from file.

        Args:
            file_path: Path to configuration file
            format: File format (auto-detected if None)

        Returns:
            Loaded configuration data

        Raises:
            ConfigPersistenceError: If loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ConfigPersistenceError(f"Configuration file not found: {file_path}")

        try:
            # Auto-detect format if not specified
            if format is None:
                format = self._detect_format(file_path)

            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Deserialize data
            data = self._deserialize_data(content, format)

            # Apply decryption if encryption is enabled
            if (
                self.enable_encryption
                and self.encryption
                and self.encryption.is_encryption_enabled()
            ):
                try:
                    data = self.encryption.decrypt_config_dict(data)
                    self.logger.debug(
                        "Applied decryption to encrypted configuration fields"
                    )
                except ConfigEncryptionError as e:
                    self.logger.warning(f"Decryption failed, loading as-is: {e}")

            self.logger.info(f"Configuration loaded successfully: {file_path}")
            return data

        except Exception as e:
            raise ConfigPersistenceError(
                f"Failed to load configuration from {file_path}: {e}"
            )

    def save_profile(
        self,
        profile_name: str,
        config_data: Union[dict[str, Any], ConfigProfileEnhanced, ApplicationConfig],
        format: ConfigFormat = ConfigFormat.JSON,
    ) -> bool:
        """Save a named configuration profile.

        Args:
            profile_name: Name of the profile
            config_data: Configuration data or profile object
            format: File format to save as

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            profile_file = self.profiles_dir / f"{profile_name}.{format.value}"

            # Handle different input types
            if isinstance(config_data, ConfigProfileEnhanced):
                data = config_data.model_dump()
            elif isinstance(config_data, ApplicationConfig):
                # Create a profile from full config
                profile = ConfigProfileEnhanced(
                    name=profile_name,
                    description="Profile created from full configuration",
                    settings=config_data.model_dump(),
                )
                data = profile.model_dump()
            else:
                # Assume it's a dict - wrap in profile structure if needed
                if "name" not in config_data:
                    profile = ConfigProfileEnhanced(
                        name=profile_name,
                        description=f"Profile for {profile_name}",
                        settings=config_data,
                    )
                    data = profile.model_dump()
                else:
                    data = dict(config_data)

            return self.save_config(data, profile_file, format)

        except Exception as e:
            self.logger.error(f"Failed to save profile '{profile_name}': {e}")
            return False

    def load_profile(
        self, profile_name: str, format: Optional[ConfigFormat] = None
    ) -> ConfigProfileEnhanced:
        """Load a named configuration profile.

        Args:
            profile_name: Name of the profile to load
            format: File format (auto-detected if None)

        Returns:
            Loaded profile object

        Raises:
            ConfigPersistenceError: If profile loading fails
        """
        try:
            # Try different format extensions if format not specified
            if format is None:
                # Try common formats
                for fmt in [ConfigFormat.JSON, ConfigFormat.YAML]:
                    profile_file = self.profiles_dir / f"{profile_name}.{fmt.value}"
                    if profile_file.exists():
                        format = fmt
                        break
                else:
                    raise ConfigPersistenceError(f"Profile '{profile_name}' not found")
            else:
                profile_file = self.profiles_dir / f"{profile_name}.{format.value}"

            data = self.load_config(profile_file, format)

            # Create ConfigProfileEnhanced object
            return ConfigProfileEnhanced(**data)

        except Exception as e:
            raise ConfigPersistenceError(
                f"Failed to load profile '{profile_name}': {e}"
            )

    def list_profiles(self) -> list[str]:
        """List all available configuration profiles.

        Returns:
            List of profile names
        """
        profiles = set()

        try:
            for file_path in self.profiles_dir.glob("*"):
                if file_path.is_file() and file_path.suffix in [
                    ".json",
                    ".yaml",
                    ".yml",
                ]:
                    profiles.add(file_path.stem)

            return sorted(profiles)

        except Exception as e:
            self.logger.error(f"Failed to list profiles: {e}")
            return []

    def delete_profile(self, profile_name: str) -> bool:
        """Delete a configuration profile.

        Args:
            profile_name: Name of the profile to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            deleted = False

            # Try all possible format extensions
            for fmt in [ConfigFormat.JSON, ConfigFormat.YAML]:
                profile_file = self.profiles_dir / f"{profile_name}.{fmt.value}"
                if profile_file.exists():
                    # Create backup before deletion
                    self._create_backup(profile_file)
                    profile_file.unlink()
                    deleted = True
                    self.logger.info(f"Deleted profile: {profile_file}")

            return deleted

        except Exception as e:
            self.logger.error(f"Failed to delete profile '{profile_name}': {e}")
            return False

    def cleanup_backups(self, max_backups: int = 10) -> int:
        """Clean up old backup files.

        Args:
            max_backups: Maximum number of backups to keep per file

        Returns:
            Number of backups removed
        """
        try:
            removed_count = 0

            # Group backups by original filename
            backup_groups = {}
            for backup_file in self.backups_dir.glob("*"):
                if backup_file.is_file():
                    # Extract original filename (remove timestamp)
                    name_parts = backup_file.stem.split("_")
                    if len(name_parts) >= 2:
                        original_name = "_".join(name_parts[:-1])
                        backup_groups.setdefault(original_name, []).append(backup_file)

            # Remove oldest backups if exceeding limit
            for original_name, backups in backup_groups.items():
                if len(backups) > max_backups:
                    # Sort by modification time (oldest first)
                    backups.sort(key=lambda f: f.stat().st_mtime)

                    # Remove oldest backups
                    for backup_file in backups[:-max_backups]:
                        backup_file.unlink()
                        removed_count += 1
                        self.logger.debug(f"Removed old backup: {backup_file}")

            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old backup files")

            return removed_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup backups: {e}")
            return 0

    def save(self, config: dict, path: str) -> bool:
        """Legacy method for backward compatibility.

        Args:
            config: Configuration data to save
            path: Path to save configuration

        Returns:
            True if saved successfully, False otherwise
        """
        return self.save_config(config, path)

    def load(self, path: str) -> dict:
        """Legacy method for backward compatibility.

        Args:
            path: Path to configuration file

        Returns:
            Loaded configuration data

        Raises:
            ConfigPersistenceError: If loading fails
        """
        return self.load_config(path)

    def enable_config_encryption(
        self,
        encryption: Optional[ConfigEncryption] = None,
        password: Optional[str] = None,
    ) -> bool:
        """Enable configuration encryption.

        Args:
            encryption: Optional ConfigEncryption instance. If None, creates a new one.
            password: Optional password for key derivation/decryption

        Returns:
            True if encryption was enabled successfully, False otherwise
        """
        try:
            if encryption:
                self.encryption = encryption
            else:
                self.encryption = ConfigEncryption()
                self.encryption.initialize(password)

            self.enable_encryption = True
            self.logger.info("Configuration encryption enabled")
            return True

        except Exception as e:
            self.logger.error(f"Failed to enable encryption: {e}")
            return False

    def disable_config_encryption(self) -> None:
        """Disable configuration encryption."""
        self.enable_encryption = False
        self.encryption = None
        self.logger.info("Configuration encryption disabled")

    def is_encryption_enabled(self) -> bool:
        """Check if encryption is enabled.

        Returns:
            True if encryption is enabled and ready
        """
        return (
            self.enable_encryption
            and self.encryption is not None
            and self.encryption.is_encryption_enabled()
        )

    def migrate_to_encrypted(
        self, file_path: Union[str, Path], password: Optional[str] = None
    ) -> bool:
        """Migrate an existing unencrypted configuration file to encrypted format.

        Args:
            file_path: Path to configuration file to migrate
            password: Password for encryption key derivation

        Returns:
            True if migration was successful, False otherwise
        """
        file_path = Path(file_path)

        try:
            if not file_path.exists():
                self.logger.error(f"Configuration file not found: {file_path}")
                return False

            # Load current configuration without encryption
            old_encryption_state = self.enable_encryption
            self.enable_encryption = False

            try:
                config_data = self.load_config(file_path)
            finally:
                self.enable_encryption = old_encryption_state

            # Enable encryption
            if not self.is_encryption_enabled():
                if not self.enable_config_encryption(password=password):
                    return False

            # Save with encryption
            backup_created = self._create_backup(file_path)
            if backup_created:
                self.logger.info(
                    f"Created backup before encryption migration: {backup_created}"
                )

            success = self.save_config(config_data, file_path, create_backup=False)
            if success:
                self.logger.info(
                    f"Successfully migrated {file_path} to encrypted format"
                )
            else:
                self.logger.error(
                    f"Failed to save encrypted configuration to {file_path}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Migration to encrypted format failed: {e}")
            return False

    def get_encryption_info(self) -> dict[str, Any]:
        """Get information about the current encryption configuration.

        Returns:
            Dictionary containing encryption status and configuration
        """
        info = {
            "encryption_enabled": self.enable_encryption,
            "encryption_ready": self.is_encryption_enabled(),
            "secure_fields": list(self.encryption.get_secure_fields())
            if self.encryption
            else [],
            "encryption_available": self.encryption is not None,
        }

        if self.encryption:
            info.update(
                {
                    "key_file": str(self.encryption.key_manager.key_file),
                    "key_file_exists": self.encryption.key_manager.key_file.exists(),
                }
            )

        return info
