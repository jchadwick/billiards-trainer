"""Configuration encryption for secure storage.

This module provides secure encryption and decryption for sensitive configuration
values using industry-standard cryptographic libraries. It supports:

- Symmetric encryption using Fernet (AES 128 in CBC mode with HMAC authentication)
- Key derivation from passwords using PBKDF2
- Secure key storage and management
- Transparent handling of sensitive configuration fields
- Backward compatibility with unencrypted configurations
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

    # Mock classes for development/testing when cryptography is not available
    class MockFernet:
        @staticmethod
        def generate_key():
            return b"mock_key_for_testing_32_bytes_long"

        def __init__(self, key):
            self.key = key

        def encrypt(self, data: bytes) -> bytes:
            return data  # Return data unchanged in mock mode

        def decrypt(self, token: bytes) -> bytes:
            return token  # Return data unchanged in mock mode

    Fernet = MockFernet

    class MockPBKDF2HMAC:
        def __init__(self, algorithm, length, salt, iterations):
            pass

        def derive(self, password: bytes) -> bytes:
            return b"mock_derived_key_32_bytes_long__"

    PBKDF2HMAC = MockPBKDF2HMAC
    hashes = None


logger = logging.getLogger(__name__)


class ConfigEncryptionError(Exception):
    """Configuration encryption related errors."""

    pass


class KeyDerivationError(ConfigEncryptionError):
    """Key derivation related errors."""

    pass


class EncryptionKeyManager:
    """Manages encryption keys and key derivation."""

    def __init__(self, key_file: Optional[Path] = None):
        """Initialize key manager.

        Args:
            key_file: Optional path to key file. If None, uses default location.
        """
        self.key_file = key_file or Path("config/.encryption_key")
        self._master_key: Optional[bytes] = None
        self._derived_keys: dict[str, bytes] = {}

    def _ensure_key_directory(self) -> None:
        """Ensure the key directory exists."""
        self.key_file.parent.mkdir(parents=True, exist_ok=True)

    def generate_master_key(self) -> bytes:
        """Generate a new master encryption key.

        Returns:
            Generated master key
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography library not available, using mock key")
            return b"mock_master_key_32_bytes_long___"

        return Fernet.generate_key()

    def derive_key_from_password(
        self, password: str, salt: Optional[bytes] = None
    ) -> bytes:
        """Derive encryption key from password using PBKDF2.

        Args:
            password: Password to derive key from
            salt: Optional salt. If None, generates a new random salt.

        Returns:
            Derived key

        Raises:
            KeyDerivationError: If key derivation fails
        """
        try:
            if not CRYPTOGRAPHY_AVAILABLE:
                logger.warning(
                    "Cryptography library not available, using mock derivation"
                )
                return b"mock_derived_key_32_bytes_long__"

            if salt is None:
                salt = os.urandom(16)  # 16 bytes salt

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 32 bytes for Fernet key
                salt=salt,
                iterations=100000,  # OWASP recommended minimum
            )

            key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
            return key

        except Exception as e:
            raise KeyDerivationError(f"Failed to derive key from password: {e}")

    def save_master_key(self, key: bytes, password: Optional[str] = None) -> None:
        """Save master key to file, optionally encrypted with password.

        Args:
            key: Master key to save
            password: Optional password to encrypt the key file

        Raises:
            ConfigEncryptionError: If saving fails
        """
        try:
            self._ensure_key_directory()

            if password:
                # Encrypt the key with password-derived key
                salt = os.urandom(16)
                derived_key = self.derive_key_from_password(password, salt)
                cipher = Fernet(derived_key)
                encrypted_key = cipher.encrypt(key)

                # Store salt + encrypted key
                key_data = {
                    "encrypted": True,
                    "salt": base64.b64encode(salt).decode("utf-8"),
                    "key": base64.b64encode(encrypted_key).decode("utf-8"),
                }
            else:
                # Store key unencrypted (not recommended for production)
                key_data = {
                    "encrypted": False,
                    "key": base64.b64encode(key).decode("utf-8"),
                }

            with open(self.key_file, "w") as f:
                json.dump(key_data, f, indent=2)

            # Set restrictive permissions
            os.chmod(self.key_file, 0o600)

            logger.info(f"Master key saved to {self.key_file}")

        except Exception as e:
            raise ConfigEncryptionError(f"Failed to save master key: {e}")

    def load_master_key(self, password: Optional[str] = None) -> bytes:
        """Load master key from file.

        Args:
            password: Password to decrypt key file if encrypted

        Returns:
            Master encryption key

        Raises:
            ConfigEncryptionError: If loading fails
        """
        try:
            if not self.key_file.exists():
                raise ConfigEncryptionError(f"Key file not found: {self.key_file}")

            with open(self.key_file) as f:
                key_data = json.load(f)

            if key_data.get("encrypted", False):
                if not password:
                    raise ConfigEncryptionError("Password required to decrypt key file")

                salt = base64.b64decode(key_data["salt"])
                encrypted_key = base64.b64decode(key_data["key"])

                derived_key = self.derive_key_from_password(password, salt)
                cipher = Fernet(derived_key)
                master_key = cipher.decrypt(encrypted_key)
            else:
                master_key = base64.b64decode(key_data["key"])

            self._master_key = master_key
            logger.info("Master key loaded successfully")
            return master_key

        except Exception as e:
            raise ConfigEncryptionError(f"Failed to load master key: {e}")

    def get_or_create_master_key(self, password: Optional[str] = None) -> bytes:
        """Get existing master key or create a new one.

        Args:
            password: Password for key file encryption

        Returns:
            Master encryption key
        """
        try:
            return self.load_master_key(password)
        except ConfigEncryptionError:
            logger.info("Creating new master key")
            key = self.generate_master_key()
            self.save_master_key(key, password)
            self._master_key = key
            return key


class ConfigEncryption:
    """Configuration encryption for secure storage.

    Provides transparent encryption and decryption of sensitive configuration values
    using Fernet symmetric encryption with key derivation and secure key management.
    """

    # Fields that should be encrypted by default
    DEFAULT_SECURE_FIELDS = {
        "jwt_secret_key",
        "api_key",
        "secret_key",
        "password",
        "token",
        "credential",
        "private_key",
        "database_password",
        "redis_password",
    }

    # Prefix to identify encrypted values
    ENCRYPTED_PREFIX = "ENCRYPTED:"

    def __init__(
        self,
        key_manager: Optional[EncryptionKeyManager] = None,
        secure_fields: Optional[list[str]] = None,
    ):
        """Initialize configuration encryption.

        Args:
            key_manager: Key manager instance. If None, creates default one.
            secure_fields: List of field names to encrypt. If None, uses defaults.
        """
        self.key_manager = key_manager or EncryptionKeyManager()
        self.secure_fields = set(secure_fields or self.DEFAULT_SECURE_FIELDS)
        self._cipher: Optional[Fernet] = None
        self._initialized = False

    def initialize(self, password: Optional[str] = None) -> None:
        """Initialize encryption with master key.

        Args:
            password: Password for key derivation/decryption

        Raises:
            ConfigEncryptionError: If initialization fails
        """
        try:
            master_key = self.key_manager.get_or_create_master_key(password)
            self._cipher = Fernet(master_key)
            self._initialized = True
            logger.info("Configuration encryption initialized")

        except Exception as e:
            raise ConfigEncryptionError(f"Failed to initialize encryption: {e}")

    def _ensure_initialized(self) -> None:
        """Ensure encryption is initialized."""
        if not self._initialized:
            raise ConfigEncryptionError(
                "Encryption not initialized. Call initialize() first."
            )

    def _is_secure_field(self, key: str) -> bool:
        """Check if a field should be encrypted based on its name.

        Args:
            key: Field name to check

        Returns:
            True if field should be encrypted
        """
        key_lower = key.lower()
        return any(secure_field in key_lower for secure_field in self.secure_fields)

    def _is_encrypted_value(self, value: str) -> bool:
        """Check if a value is already encrypted.

        Args:
            value: Value to check

        Returns:
            True if value appears to be encrypted
        """
        return isinstance(value, str) and value.startswith(self.ENCRYPTED_PREFIX)

    def encrypt_value(self, value: str) -> str:
        """Encrypt a single configuration value.

        Args:
            value: Value to encrypt

        Returns:
            Encrypted value with prefix

        Raises:
            ConfigEncryptionError: If encryption fails
        """
        self._ensure_initialized()

        if not isinstance(value, str):
            raise ConfigEncryptionError("Only string values can be encrypted")

        if self._is_encrypted_value(value):
            return value  # Already encrypted

        try:
            encrypted_bytes = self._cipher.encrypt(value.encode("utf-8"))
            encrypted_b64 = base64.b64encode(encrypted_bytes).decode("utf-8")
            return f"{self.ENCRYPTED_PREFIX}{encrypted_b64}"

        except Exception as e:
            raise ConfigEncryptionError(f"Failed to encrypt value: {e}")

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a single configuration value.

        Args:
            encrypted_value: Encrypted value to decrypt

        Returns:
            Decrypted plain text value

        Raises:
            ConfigEncryptionError: If decryption fails
        """
        self._ensure_initialized()

        if not self._is_encrypted_value(encrypted_value):
            return encrypted_value  # Not encrypted, return as-is

        try:
            # Remove prefix and decode
            encrypted_b64 = encrypted_value[len(self.ENCRYPTED_PREFIX) :]
            encrypted_bytes = base64.b64decode(encrypted_b64)

            # Decrypt
            decrypted_bytes = self._cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode("utf-8")

        except Exception as e:
            raise ConfigEncryptionError(f"Failed to decrypt value: {e}")

    def add_secure_field(self, field_name: str) -> None:
        """Add a field name to the list of fields that should be encrypted.

        Args:
            field_name: Name of field to encrypt
        """
        self.secure_fields.add(field_name.lower())
        logger.debug(f"Added secure field: {field_name}")

    def remove_secure_field(self, field_name: str) -> None:
        """Remove a field name from the list of fields that should be encrypted.

        Args:
            field_name: Name of field to stop encrypting
        """
        self.secure_fields.discard(field_name.lower())
        logger.debug(f"Removed secure field: {field_name}")

    def encrypt(self, data: str, key: Optional[str] = None) -> str:
        """Encrypt configuration data (legacy method for backward compatibility).

        Args:
            data: Data to encrypt
            key: Legacy key parameter (ignored, uses managed keys)

        Returns:
            Encrypted data
        """
        return self.encrypt_value(data)

    def decrypt(self, encrypted_data: str, key: Optional[str] = None) -> str:
        """Decrypt configuration data (legacy method for backward compatibility).

        Args:
            encrypted_data: Data to decrypt
            key: Legacy key parameter (ignored, uses managed keys)

        Returns:
            Decrypted data
        """
        return self.decrypt_value(encrypted_data)

    def encrypt_config_dict(
        self, config_dict: dict[str, Any], path_prefix: str = ""
    ) -> dict[str, Any]:
        """Recursively encrypt sensitive fields in a configuration dictionary.

        Args:
            config_dict: Configuration dictionary to encrypt
            path_prefix: Current path prefix for nested keys

        Returns:
            Dictionary with sensitive fields encrypted
        """
        if not isinstance(config_dict, dict):
            return config_dict

        encrypted_dict = {}
        for key, value in config_dict.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            if isinstance(value, dict):
                # Recursively process nested dictionaries
                encrypted_dict[key] = self.encrypt_config_dict(value, current_path)
            elif isinstance(value, list):
                # Process lists
                encrypted_dict[key] = self._encrypt_config_list(value, current_path)
            elif isinstance(value, str) and self._is_secure_field(key):
                # Encrypt sensitive string fields
                try:
                    encrypted_dict[key] = self.encrypt_value(value)
                    logger.debug(f"Encrypted field: {current_path}")
                except ConfigEncryptionError as e:
                    logger.warning(f"Failed to encrypt field {current_path}: {e}")
                    encrypted_dict[key] = value  # Keep original if encryption fails
            else:
                # Keep non-sensitive fields as-is
                encrypted_dict[key] = value

        return encrypted_dict

    def decrypt_config_dict(
        self, config_dict: dict[str, Any], path_prefix: str = ""
    ) -> dict[str, Any]:
        """Recursively decrypt encrypted fields in a configuration dictionary.

        Args:
            config_dict: Configuration dictionary to decrypt
            path_prefix: Current path prefix for nested keys

        Returns:
            Dictionary with encrypted fields decrypted
        """
        if not isinstance(config_dict, dict):
            return config_dict

        decrypted_dict = {}
        for key, value in config_dict.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            if isinstance(value, dict):
                # Recursively process nested dictionaries
                decrypted_dict[key] = self.decrypt_config_dict(value, current_path)
            elif isinstance(value, list):
                # Process lists
                decrypted_dict[key] = self._decrypt_config_list(value, current_path)
            elif isinstance(value, str) and self._is_encrypted_value(value):
                # Decrypt encrypted string fields
                try:
                    decrypted_dict[key] = self.decrypt_value(value)
                    logger.debug(f"Decrypted field: {current_path}")
                except ConfigEncryptionError as e:
                    logger.warning(f"Failed to decrypt field {current_path}: {e}")
                    decrypted_dict[key] = value  # Keep original if decryption fails
            else:
                # Keep non-encrypted fields as-is
                decrypted_dict[key] = value

        return decrypted_dict

    def _encrypt_config_list(
        self, config_list: list[Any], path_prefix: str
    ) -> list[Any]:
        """Encrypt items in a configuration list.

        Args:
            config_list: List to process
            path_prefix: Current path prefix

        Returns:
            List with encrypted items
        """
        encrypted_list = []
        for i, item in enumerate(config_list):
            item_path = f"{path_prefix}[{i}]"

            if isinstance(item, dict):
                encrypted_list.append(self.encrypt_config_dict(item, item_path))
            elif isinstance(item, list):
                encrypted_list.append(self._encrypt_config_list(item, item_path))
            else:
                encrypted_list.append(item)

        return encrypted_list

    def _decrypt_config_list(
        self, config_list: list[Any], path_prefix: str
    ) -> list[Any]:
        """Decrypt items in a configuration list.

        Args:
            config_list: List to process
            path_prefix: Current path prefix

        Returns:
            List with decrypted items
        """
        decrypted_list = []
        for i, item in enumerate(config_list):
            item_path = f"{path_prefix}[{i}]"

            if isinstance(item, dict):
                decrypted_list.append(self.decrypt_config_dict(item, item_path))
            elif isinstance(item, list):
                decrypted_list.append(self._decrypt_config_list(item, item_path))
            else:
                decrypted_list.append(item)

        return decrypted_list

    def is_encryption_enabled(self) -> bool:
        """Check if encryption is enabled and initialized.

        Returns:
            True if encryption is enabled and ready to use
        """
        return self._initialized

    def get_secure_fields(self) -> set[str]:
        """Get the current set of secure field names.

        Returns:
            Set of field names that will be encrypted
        """
        return self.secure_fields.copy()

    def rotate_keys(self, new_password: Optional[str] = None) -> None:
        """Rotate encryption keys by generating a new master key.

        Args:
            new_password: New password for key file encryption

        Raises:
            ConfigEncryptionError: If key rotation fails
        """
        try:
            logger.info("Starting key rotation")

            # Generate new master key
            new_master_key = self.key_manager.generate_master_key()

            # Save new key
            self.key_manager.save_master_key(new_master_key, new_password)

            # Update cipher with new key
            self._cipher = Fernet(new_master_key)
            self.key_manager._master_key = new_master_key

            logger.info("Key rotation completed successfully")

        except Exception as e:
            raise ConfigEncryptionError(f"Key rotation failed: {e}")

    def export_encrypted_config(self, config_dict: dict[str, Any]) -> str:
        """Export configuration with sensitive fields encrypted as JSON string.

        Args:
            config_dict: Configuration dictionary to export

        Returns:
            JSON string with encrypted sensitive fields
        """
        encrypted_dict = self.encrypt_config_dict(config_dict)
        return json.dumps(encrypted_dict, indent=2, default=str, ensure_ascii=False)

    def import_encrypted_config(self, config_json: str) -> dict[str, Any]:
        """Import configuration from JSON string with encrypted fields.

        Args:
            config_json: JSON string containing encrypted configuration

        Returns:
            Configuration dictionary with decrypted fields
        """
        config_dict = json.loads(config_json)
        return self.decrypt_config_dict(config_dict)
