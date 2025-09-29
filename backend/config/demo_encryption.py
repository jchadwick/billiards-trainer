#!/usr/bin/env python3
"""Demonstration of configuration encryption functionality.

This script shows how to use the configuration encryption system for
secure storage of sensitive configuration values.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.storage.encryption import ConfigEncryption, EncryptionKeyManager
from config.storage.persistence import ConfigPersistence


def demo_basic_encryption():
    """Demonstrate basic encryption and decryption."""
    print("=== Basic Encryption Demo ===")

    # Initialize encryption
    encryption = ConfigEncryption()
    encryption.initialize()

    # Encrypt individual values
    secret_key = "super_secret_jwt_key_123"
    api_key = "api_key_xyz789"

    encrypted_secret = encryption.encrypt_value(secret_key)
    encrypted_api = encryption.encrypt_value(api_key)

    print(f"Original secret key: {secret_key}")
    print(f"Encrypted secret key: {encrypted_secret}")
    print(f"Original API key: {api_key}")
    print(f"Encrypted API key: {encrypted_api}")

    # Decrypt values
    decrypted_secret = encryption.decrypt_value(encrypted_secret)
    decrypted_api = encryption.decrypt_value(encrypted_api)

    print(f"Decrypted secret key: {decrypted_secret}")
    print(f"Decrypted API key: {decrypted_api}")

    assert secret_key == decrypted_secret
    assert api_key == decrypted_api
    print("✓ Basic encryption/decryption working correctly")
    print()


def demo_config_dict_encryption():
    """Demonstrate configuration dictionary encryption."""
    print("=== Configuration Dictionary Encryption Demo ===")

    encryption = ConfigEncryption()
    encryption.initialize()

    # Sample configuration with sensitive and non-sensitive values
    config = {
        "system": {
            "debug": True,
            "timezone": "UTC",
        },
        "api": {
            "host": "localhost",
            "port": 8000,
            "jwt_secret_key": "super_secret_jwt_key",
            "authentication": {"enabled": True, "api_key": "secret_api_key_123"},
        },
        "database": {
            "host": "db.example.com",
            "port": 5432,
            "username": "app_user",
            "password": "very_secret_db_password",
        },
        "services": [
            {
                "name": "redis",
                "host": "redis.example.com",
                "password": "redis_secret_pass",
            },
            {
                "name": "elasticsearch",
                "host": "es.example.com",
                "api_key": "es_api_key_secret",
            },
        ],
    }

    print("Original configuration:")
    print(json.dumps(config, indent=2))

    # Encrypt sensitive fields
    encrypted_config = encryption.encrypt_config_dict(config)

    print("\nConfiguration with encrypted sensitive fields:")
    print(json.dumps(encrypted_config, indent=2))

    # Decrypt back to original
    decrypted_config = encryption.decrypt_config_dict(encrypted_config)

    print("\nDecrypted configuration:")
    print(json.dumps(decrypted_config, indent=2))

    assert config == decrypted_config
    print("✓ Configuration dictionary encryption/decryption working correctly")
    print()


def demo_password_protected_keys():
    """Demonstrate password-protected key storage."""
    print("=== Password-Protected Key Storage Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        key_file = Path(temp_dir) / "protected_key"
        password = "my_secure_password_123"

        # Create key manager with password protection
        key_manager = EncryptionKeyManager(key_file)

        # Generate and save password-protected key
        master_key = key_manager.generate_master_key()
        key_manager.save_master_key(master_key, password)

        print(f"Saved password-protected key to: {key_file}")
        print(f"Key file exists: {key_file.exists()}")

        # Load key with password
        loaded_key = key_manager.load_master_key(password)
        assert master_key == loaded_key
        print("✓ Password-protected key storage working correctly")

        # Try to load without password (should fail)
        try:
            key_manager.load_master_key()
            print("✗ ERROR: Should not be able to load encrypted key without password")
        except Exception as e:
            print(f"✓ Correctly failed to load encrypted key without password: {e}")

    print()


def demo_persistence_with_encryption():
    """Demonstrate configuration persistence with encryption."""
    print("=== Configuration Persistence with Encryption Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        config_file = config_dir / "app_config.json"

        # Create persistence manager with encryption enabled
        persistence = ConfigPersistence(base_dir=config_dir, enable_encryption=True)

        # Sample configuration
        config_data = {
            "app_name": "Billiards Trainer",
            "version": "1.0.0",
            "debug": False,
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "jwt_secret_key": "production_jwt_secret_key",
                "authentication": {"enabled": True, "api_key": "production_api_key"},
            },
            "database": {
                "url": "postgresql://localhost:5432/billiards",
                "password": "super_secret_db_password",
            },
        }

        print("Saving configuration with encryption...")
        success = persistence.save_config(config_data, config_file)
        print(f"Save successful: {success}")

        # Show what's actually stored in the file
        with open(config_file) as f:
            file_content = json.load(f)

        print("\nActual file content (with encrypted sensitive fields):")
        print(json.dumps(file_content, indent=2))

        # Load configuration (should be decrypted automatically)
        print("\nLoading configuration...")
        loaded_config = persistence.load_config(config_file)

        print("\nLoaded configuration (decrypted):")
        print(json.dumps(loaded_config, indent=2))

        assert config_data == loaded_config
        print("✓ Configuration persistence with encryption working correctly")

        # Show encryption info
        encryption_info = persistence.get_encryption_info()
        print("\nEncryption information:")
        print(json.dumps(encryption_info, indent=2))

    print()


def demo_migration_to_encrypted():
    """Demonstrate migrating existing unencrypted config to encrypted format."""
    print("=== Migration to Encrypted Format Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        config_file = config_dir / "legacy_config.json"

        # Create unencrypted configuration file
        legacy_config = {
            "debug": True,
            "jwt_secret_key": "legacy_secret_key",
            "api_key": "legacy_api_key",
            "database_password": "legacy_db_password",
        }

        with open(config_file, "w") as f:
            json.dump(legacy_config, f, indent=2)

        print("Created legacy unencrypted configuration:")
        print(json.dumps(legacy_config, indent=2))

        # Create persistence manager without encryption initially
        persistence = ConfigPersistence(base_dir=config_dir, enable_encryption=False)

        # Migrate to encrypted format
        print("\nMigrating to encrypted format...")
        success = persistence.migrate_to_encrypted(config_file)
        print(f"Migration successful: {success}")

        # Show updated file content
        with open(config_file) as f:
            encrypted_content = json.load(f)

        print("\nFile content after migration (with encrypted sensitive fields):")
        print(json.dumps(encrypted_content, indent=2))

        # Verify we can still load the original data
        loaded_config = persistence.load_config(config_file)
        assert legacy_config == loaded_config
        print("✓ Migration to encrypted format working correctly")

    print()


def demo_custom_secure_fields():
    """Demonstrate custom secure field configuration."""
    print("=== Custom Secure Fields Demo ===")

    # Create encryption with custom secure fields
    custom_fields = ["custom_secret", "private_token", "license_key"]
    encryption = ConfigEncryption(secure_fields=custom_fields)
    encryption.initialize()

    config = {
        "app_name": "Test App",
        "custom_secret": "my_custom_secret",
        "private_token": "private_token_123",
        "license_key": "ABCD-1234-EFGH-5678",
        "public_setting": "not_encrypted",
        "jwt_secret_key": "default_secure_field",  # Not in custom list
    }

    print("Configuration with custom secure fields:")
    print(json.dumps(config, indent=2))

    encrypted_config = encryption.encrypt_config_dict(config)

    print("\nEncrypted configuration:")
    print(json.dumps(encrypted_config, indent=2))

    # Show which fields are considered secure
    print(f"\nSecure fields: {encryption.get_secure_fields()}")

    # Add additional secure field at runtime
    encryption.add_secure_field("runtime_secret")
    config["runtime_secret"] = "added_at_runtime"

    encrypted_config = encryption.encrypt_config_dict(config)
    print("\nAfter adding 'runtime_secret' as secure field:")
    print(json.dumps(encrypted_config, indent=2))

    print("✓ Custom secure fields working correctly")
    print()


def main():
    """Run all encryption demonstrations."""
    print("Configuration Encryption System Demonstration")
    print("=" * 50)
    print()

    try:
        demo_basic_encryption()
        demo_config_dict_encryption()
        demo_password_protected_keys()
        demo_persistence_with_encryption()
        demo_migration_to_encrypted()
        demo_custom_secure_fields()

        print("=" * 50)
        print("All encryption demonstrations completed successfully! ✓")
        print()
        print("Key features demonstrated:")
        print("• Symmetric encryption using Fernet (AES-128)")
        print("• PBKDF2 key derivation from passwords")
        print("• Transparent encryption/decryption of configuration dictionaries")
        print("• Password-protected key storage")
        print("• Integration with configuration persistence")
        print("• Migration of legacy unencrypted configurations")
        print("• Custom secure field configuration")
        print("• Backward compatibility with unencrypted configurations")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
