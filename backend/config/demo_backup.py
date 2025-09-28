#!/usr/bin/env python3
"""Demonstration of the configuration backup and restore system."""

import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import ConfigurationModule


def main():
    """Demonstrate the backup and restore functionality."""
    print("Configuration Backup and Restore System Demo")
    print("=" * 50)

    # Initialize configuration module with a temporary directory
    config_dir = Path("demo_config")
    config_module = ConfigurationModule(config_dir=config_dir, enable_hot_reload=False)

    try:
        # 1. Set initial configuration
        print("\n1. Setting initial configuration...")
        config_module.set("app.name", "billiards-trainer", persist=True)
        config_module.set("app.version", "1.0.0", persist=True)
        config_module.set("api.port", 8000, persist=True)
        config_module.set("vision.camera.device_id", 0, persist=True)

        # Show current configuration
        print("Current configuration:")
        for key, value in config_module.get_all().items():
            if not key.startswith("logging.") and not key.startswith("projector."):
                print(f"  {key} = {value}")

        # 2. Create a manual backup
        print("\n2. Creating manual backup...")
        backup_path = config_module.create_backup(
            "initial_config",
            description="Initial application configuration",
            tags=["demo", "initial"],
        )
        print(f"Backup created: {backup_path}")

        # 3. Make some changes
        print("\n3. Making configuration changes...")
        config_module.set("app.version", "2.0.0", persist=True)
        config_module.set("api.port", 9000, persist=True)
        config_module.set("vision.camera.device_id", 1, persist=True)
        config_module.set("new.feature", "enabled", persist=True)

        print("Changed configuration:")
        for key, value in config_module.get_all().items():
            if not key.startswith("logging.") and not key.startswith("projector."):
                print(f"  {key} = {value}")

        # 4. Create automatic backup before major changes
        print("\n4. Creating automatic backup...")
        auto_backup_path = config_module.create_auto_backup("before_major_changes")
        print(f"Auto backup created: {auto_backup_path}")

        # 5. List all backups
        print("\n5. Listing all backups...")
        backups = config_module.list_config_backups()
        for backup in backups:
            backup_type = "Auto" if backup.auto_backup else "Manual"
            print(
                f"  {backup_type}: {backup.name} ({backup.creation_date.strftime('%Y-%m-%d %H:%M:%S')})"
            )
            if backup.description:
                print(f"    Description: {backup.description}")
            if backup.tags:
                print(f"    Tags: {', '.join(backup.tags)}")

        # 6. Restore from backup
        print("\n6. Restoring from 'initial_config' backup...")
        success = config_module.restore_backup("initial_config")
        if success:
            print("Backup restored successfully!")
            print("Restored configuration:")
            for key, value in config_module.get_all().items():
                if not key.startswith("logging.") and not key.startswith("projector."):
                    print(f"  {key} = {value}")
        else:
            print("Failed to restore backup")

        # 7. Verify backup integrity
        print("\n7. Verifying backup integrity...")
        for backup in config_module.list_config_backups():
            is_valid = config_module.verify_backup(backup.name)
            status = "✓ Valid" if is_valid else "✗ Corrupted"
            print(f"  {backup.name}: {status}")

        # 8. Get backup information
        print("\n8. Backup metadata...")
        backup_info = config_module.get_backup_info("initial_config")
        if backup_info:
            print(f"  Name: {backup_info.name}")
            print(f"  Size: {backup_info.size_bytes} bytes")
            print(f"  Checksum: {backup_info.checksum[:16]}...")
            print(f"  Config keys: {backup_info.config_keys_count}")

        # 9. Export backup
        print("\n9. Exporting backup...")
        export_path = Path("exported_backup.json")
        export_success = config_module.export_backup("initial_config", export_path)
        if export_success:
            print(f"Backup exported to: {export_path}")
            print(f"Export size: {export_path.stat().st_size} bytes")

        # 10. Clean up old backups
        print("\n10. Cleaning up old backups...")
        cleaned_count = config_module.cleanup_old_config_backups(
            max_age_days=0, keep_manual=True
        )
        print(f"Cleaned up {cleaned_count} old automatic backups")

        print("\nDemo completed successfully!")

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup demo files
        import shutil

        if config_dir.exists():
            shutil.rmtree(config_dir)
        export_path = Path("exported_backup.json")
        if export_path.exists():
            export_path.unlink()
        print("\nDemo files cleaned up.")


if __name__ == "__main__":
    main()
