"""Tests for configuration encryption functionality."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..storage.encryption import (
    ConfigEncryption,
    ConfigEncryptionError,
    EncryptionKeyManager,
    KeyDerivationError,
)
from ..storage.persistence import ConfigPersistence


class TestEncryptionKeyManager(unittest.TestCase):
    """Test cases for EncryptionKeyManager."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.key_file = self.temp_dir / "test_key"
        self.key_manager = EncryptionKeyManager(self.key_file)

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_master_key(self):
        """Test master key generation."""
        key = self.key_manager.generate_master_key()
        self.assertIsInstance(key, bytes)
        self.assertEqual(len(key), 44)  # Base64 encoded 32-byte key

    def test_save_and_load_unencrypted_key(self):
        """Test saving and loading unencrypted key."""
        original_key = self.key_manager.generate_master_key()

        # Save key without password
        self.key_manager.save_master_key(original_key)
        self.assertTrue(self.key_file.exists())

        # Load key
        loaded_key = self.key_manager.load_master_key()
        self.assertEqual(original_key, loaded_key)

    def test_save_and_load_encrypted_key(self):
        """Test saving and loading password-encrypted key."""
        original_key = self.key_manager.generate_master_key()
        password = "test_password_123"

        # Save key with password
        self.key_manager.save_master_key(original_key, password)
        self.assertTrue(self.key_file.exists())

        # Load key with password
        loaded_key = self.key_manager.load_master_key(password)
        self.assertEqual(original_key, loaded_key)

    def test_load_encrypted_key_without_password_fails(self):
        """Test that loading encrypted key without password fails."""
        original_key = self.key_manager.generate_master_key()
        password = "test_password_123"

        # Save key with password
        self.key_manager.save_master_key(original_key, password)

        # Try to load without password
        with self.assertRaises(ConfigEncryptionError):
            self.key_manager.load_master_key()

    def test_get_or_create_master_key(self):
        """Test get_or_create_master_key functionality."""
        # First call should create new key
        key1 = self.key_manager.get_or_create_master_key()
        self.assertTrue(self.key_file.exists())

        # Second call should load existing key
        key2 = self.key_manager.get_or_create_master_key()
        self.assertEqual(key1, key2)

    def test_derive_key_from_password(self):
        """Test key derivation from password."""
        password = "test_password"
        salt = b"test_salt_16byte"

        # Derive key twice with same inputs
        key1 = self.key_manager.derive_key_from_password(password, salt)
        key2 = self.key_manager.derive_key_from_password(password, salt)

        self.assertEqual(key1, key2)
        self.assertIsInstance(key1, bytes)

    def test_derive_key_different_passwords(self):
        """Test that different passwords produce different keys."""
        salt = b"test_salt_16byte"

        key1 = self.key_manager.derive_key_from_password("password1", salt)
        key2 = self.key_manager.derive_key_from_password("password2", salt)

        self.assertNotEqual(key1, key2)

    def test_key_file_permissions(self):
        """Test that key file has restrictive permissions."""
        key = self.key_manager.generate_master_key()
        self.key_manager.save_master_key(key)

        # Check file permissions (600 = rw--------)
        stat_info = os.stat(self.key_file)
        permissions = stat_info.st_mode & 0o777
        self.assertEqual(permissions, 0o600)


class TestConfigEncryption(unittest.TestCase):
    """Test cases for ConfigEncryption."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        key_manager = EncryptionKeyManager(self.temp_dir / "test_key")
        self.encryption = ConfigEncryption(key_manager=key_manager)
        self.encryption.initialize()

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_encrypt_decrypt_value(self):
        """Test basic value encryption and decryption."""
        original_value = "secret_password_123"

        # Encrypt
        encrypted = self.encryption.encrypt_value(original_value)
        self.assertNotEqual(original_value, encrypted)
        self.assertTrue(encrypted.startswith(self.encryption.ENCRYPTED_PREFIX))

        # Decrypt
        decrypted = self.encryption.decrypt_value(encrypted)
        self.assertEqual(original_value, decrypted)

    def test_encrypt_already_encrypted_value(self):
        """Test that encrypting already encrypted value returns same value."""
        original_value = "secret_password_123"
        encrypted_once = self.encryption.encrypt_value(original_value)
        encrypted_twice = self.encryption.encrypt_value(encrypted_once)

        self.assertEqual(encrypted_once, encrypted_twice)

    def test_decrypt_unencrypted_value(self):
        """Test that decrypting unencrypted value returns same value."""
        unencrypted_value = "not_encrypted"
        result = self.encryption.decrypt_value(unencrypted_value)

        self.assertEqual(unencrypted_value, result)

    def test_is_secure_field(self):
        """Test secure field identification."""
        # Should be encrypted
        self.assertTrue(self.encryption._is_secure_field("jwt_secret_key"))
        self.assertTrue(self.encryption._is_secure_field("password"))
        self.assertTrue(self.encryption._is_secure_field("api_key"))
        self.assertTrue(self.encryption._is_secure_field("database_password"))

        # Should not be encrypted
        self.assertFalse(self.encryption._is_secure_field("debug"))
        self.assertFalse(self.encryption._is_secure_field("port"))
        self.assertFalse(self.encryption._is_secure_field("timeout"))

    def test_add_remove_secure_fields(self):
        """Test adding and removing secure fields."""
        # Add new secure field
        self.encryption.add_secure_field("custom_secret")
        self.assertTrue(self.encryption._is_secure_field("custom_secret"))

        # Remove secure field
        self.encryption.remove_secure_field("custom_secret")
        self.assertFalse(self.encryption._is_secure_field("custom_secret"))

    def test_encrypt_config_dict_simple(self):
        """Test encrypting a simple configuration dictionary."""
        config = {
            "debug": True,
            "port": 8000,
            "jwt_secret_key": "super_secret_key",
            "database_password": "db_password_123"
        }

        encrypted_config = self.encryption.encrypt_config_dict(config)

        # Non-sensitive fields should remain unchanged
        self.assertEqual(encrypted_config["debug"], True)
        self.assertEqual(encrypted_config["port"], 8000)

        # Sensitive fields should be encrypted
        self.assertNotEqual(encrypted_config["jwt_secret_key"], "super_secret_key")
        self.assertNotEqual(encrypted_config["database_password"], "db_password_123")
        self.assertTrue(encrypted_config["jwt_secret_key"].startswith(self.encryption.ENCRYPTED_PREFIX))
        self.assertTrue(encrypted_config["database_password"].startswith(self.encryption.ENCRYPTED_PREFIX))

    def test_encrypt_config_dict_nested(self):
        """Test encrypting nested configuration dictionary."""
        config = {
            "app": {
                "debug": False,
                "secret_key": "app_secret"
            },
            "database": {
                "host": "localhost",
                "password": "db_secret"
            }
        }

        encrypted_config = self.encryption.encrypt_config_dict(config)

        # Check nested encryption
        self.assertEqual(encrypted_config["app"]["debug"], False)
        self.assertNotEqual(encrypted_config["app"]["secret_key"], "app_secret")
        self.assertEqual(encrypted_config["database"]["host"], "localhost")
        self.assertNotEqual(encrypted_config["database"]["password"], "db_secret")

    def test_decrypt_config_dict(self):
        """Test decrypting configuration dictionary."""
        config = {
            "debug": True,
            "jwt_secret_key": "super_secret_key",
            "api_key": "api_secret_123"
        }

        # Encrypt then decrypt
        encrypted_config = self.encryption.encrypt_config_dict(config)
        decrypted_config = self.encryption.decrypt_config_dict(encrypted_config)

        self.assertEqual(config, decrypted_config)

    def test_encrypt_config_with_list(self):
        """Test encrypting configuration with lists."""
        config = {
            "servers": [
                {"host": "server1", "password": "pass1"},
                {"host": "server2", "password": "pass2"}
            ],
            "debug": True
        }

        encrypted_config = self.encryption.encrypt_config_dict(config)

        # Check that passwords in list items are encrypted
        self.assertEqual(encrypted_config["servers"][0]["host"], "server1")
        self.assertNotEqual(encrypted_config["servers"][0]["password"], "pass1")
        self.assertTrue(encrypted_config["servers"][0]["password"].startswith(self.encryption.ENCRYPTED_PREFIX))

    def test_export_import_encrypted_config(self):
        """Test exporting and importing encrypted configuration as JSON."""
        config = {
            "debug": False,
            "jwt_secret_key": "secret_key_123",
            "port": 8000
        }

        # Export to JSON
        json_string = self.encryption.export_encrypted_config(config)
        self.assertIsInstance(json_string, str)

        # Verify JSON contains encrypted data
        json_data = json.loads(json_string)
        self.assertTrue(json_data["jwt_secret_key"].startswith(self.encryption.ENCRYPTED_PREFIX))

        # Import from JSON
        imported_config = self.encryption.import_encrypted_config(json_string)
        self.assertEqual(config, imported_config)

    def test_encryption_not_initialized(self):
        """Test that operations fail when encryption is not initialized."""
        uninitialized_encryption = ConfigEncryption()

        with self.assertRaises(ConfigEncryptionError):
            uninitialized_encryption.encrypt_value("test")

        with self.assertRaises(ConfigEncryptionError):
            uninitialized_encryption.decrypt_value("test")

    def test_rotate_keys(self):
        """Test key rotation functionality."""
        original_value = "test_secret"

        # Encrypt with original key
        encrypted_original = self.encryption.encrypt_value(original_value)

        # Rotate keys
        self.encryption.rotate_keys()

        # Value encrypted with new key should be different
        encrypted_new = self.encryption.encrypt_value(original_value)
        self.assertNotEqual(encrypted_original, encrypted_new)

        # But should decrypt to same value
        decrypted_new = self.encryption.decrypt_value(encrypted_new)
        self.assertEqual(original_value, decrypted_new)

    def test_legacy_methods_compatibility(self):
        """Test legacy encrypt/decrypt methods for backward compatibility."""
        test_data = "legacy_secret"

        # Test legacy methods
        encrypted = self.encryption.encrypt(test_data, "ignored_key")
        decrypted = self.encryption.decrypt(encrypted, "ignored_key")

        self.assertEqual(test_data, decrypted)


class TestConfigPersistenceWithEncryption(unittest.TestCase):
    """Test cases for ConfigPersistence with encryption enabled."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_config.json"

        # Create persistence with encryption enabled
        self.persistence = ConfigPersistence(
            base_dir=self.temp_dir,
            enable_encryption=True
        )

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_encrypted_config(self):
        """Test saving and loading configuration with encryption."""
        config_data = {
            "debug": True,
            "port": 8000,
            "jwt_secret_key": "super_secret_key_123",
            "api_key": "api_secret_456"
        }

        # Save configuration
        success = self.persistence.save_config(config_data, self.config_file)
        self.assertTrue(success)
        self.assertTrue(self.config_file.exists())

        # Verify file contains encrypted data
        with open(self.config_file, 'r') as f:
            file_content = json.load(f)

        # Sensitive fields should be encrypted in file
        self.assertTrue(file_content["jwt_secret_key"].startswith("ENCRYPTED:"))
        self.assertTrue(file_content["api_key"].startswith("ENCRYPTED:"))
        # Non-sensitive fields should not be encrypted
        self.assertEqual(file_content["debug"], True)
        self.assertEqual(file_content["port"], 8000)

        # Load configuration (should be decrypted)
        loaded_config = self.persistence.load_config(self.config_file)
        self.assertEqual(config_data, loaded_config)

    def test_migration_to_encrypted(self):
        """Test migrating unencrypted configuration to encrypted format."""
        config_data = {
            "debug": True,
            "jwt_secret_key": "secret_to_encrypt",
            "port": 8000
        }

        # Save without encryption first
        self.persistence.disable_config_encryption()
        self.persistence.save_config(config_data, self.config_file)

        # Verify file contains unencrypted data
        with open(self.config_file, 'r') as f:
            file_content = json.load(f)
        self.assertEqual(file_content["jwt_secret_key"], "secret_to_encrypt")

        # Migrate to encrypted format
        success = self.persistence.migrate_to_encrypted(self.config_file)
        self.assertTrue(success)

        # Verify file now contains encrypted data
        with open(self.config_file, 'r') as f:
            file_content = json.load(f)
        self.assertTrue(file_content["jwt_secret_key"].startswith("ENCRYPTED:"))

        # Verify we can still load the correct data
        loaded_config = self.persistence.load_config(self.config_file)
        self.assertEqual(config_data, loaded_config)

    def test_encryption_info(self):
        """Test getting encryption information."""
        info = self.persistence.get_encryption_info()

        self.assertIsInstance(info, dict)
        self.assertIn("encryption_enabled", info)
        self.assertIn("encryption_ready", info)
        self.assertIn("secure_fields", info)
        self.assertIn("key_file", info)

    def test_backward_compatibility_load(self):
        """Test loading unencrypted config with encryption enabled."""
        config_data = {
            "debug": True,
            "jwt_secret_key": "unencrypted_secret",
            "port": 8000
        }

        # Create unencrypted file manually
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)

        # Load with encryption enabled (should work)
        loaded_config = self.persistence.load_config(self.config_file)
        self.assertEqual(config_data, loaded_config)

    def test_encryption_error_handling(self):
        """Test error handling when encryption fails."""
        # Test with invalid encryption state
        self.persistence.encryption = None

        config_data = {"jwt_secret_key": "secret"}

        # Should save without encryption when encryption fails
        success = self.persistence.save_config(config_data, self.config_file)
        self.assertTrue(success)  # Should still succeed, just without encryption


if __name__ == "__main__":
    unittest.main()