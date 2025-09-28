#!/usr/bin/env python3
"""Backup and recovery system for Billiards Trainer."""

import argparse
import asyncio
import datetime
import gzip
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BackupManager:
    """Manages backup and recovery operations."""

    def __init__(self, system_dir: str, backup_dir: str):
        """Initialize backup manager.

        Args:
            system_dir: Main system directory to backup
            backup_dir: Directory to store backups
        """
        self.system_dir = Path(system_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Backup Manager initialized")
        logger.info(f"System dir: {self.system_dir}")
        logger.info(f"Backup dir: {self.backup_dir}")

    async def create_full_backup(self, name: Optional[str] = None) -> str:
        """Create a complete system backup.

        Args:
            name: Optional backup name (timestamp used if None)

        Returns:
            Path to created backup file
        """
        try:
            # Generate backup name
            if name is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"full_backup_{timestamp}"

            logger.info(f"Creating full backup: {name}")

            # Create temporary directory for backup
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                backup_root = temp_path / name

                # Create backup structure
                await self._backup_code(backup_root / "code")
                await self._backup_configuration(backup_root / "config")
                await self._backup_data(backup_root / "data")
                await self._backup_logs(backup_root / "logs")
                await self._backup_system_info(backup_root / "system")

                # Create manifest
                manifest = await self._create_manifest()
                with open(backup_root / "manifest.json", 'w') as f:
                    json.dump(manifest, f, indent=2)

                # Create compressed archive
                archive_path = self.backup_dir / f"{name}.tar.gz"
                await self._create_archive(backup_root, archive_path)

                logger.info(f"Full backup created: {archive_path}")
                return str(archive_path)

        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            raise

    async def create_data_backup(self, name: Optional[str] = None) -> str:
        """Create a data-only backup.

        Args:
            name: Optional backup name

        Returns:
            Path to created backup file
        """
        try:
            if name is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"data_backup_{timestamp}"

            logger.info(f"Creating data backup: {name}")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                backup_root = temp_path / name

                # Backup only data and configuration
                await self._backup_configuration(backup_root / "config")
                await self._backup_data(backup_root / "data")

                # Create manifest
                manifest = {
                    "type": "data",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "version": await self._get_system_version()
                }
                with open(backup_root / "manifest.json", 'w') as f:
                    json.dump(manifest, f, indent=2)

                # Create archive
                archive_path = self.backup_dir / f"{name}.tar.gz"
                await self._create_archive(backup_root, archive_path)

                logger.info(f"Data backup created: {archive_path}")
                return str(archive_path)

        except Exception as e:
            logger.error(f"Data backup failed: {e}")
            raise

    async def restore_backup(self, backup_path: str,
                           restore_data: bool = True,
                           restore_config: bool = True,
                           restore_code: bool = False) -> bool:
        """Restore from backup.

        Args:
            backup_path: Path to backup file
            restore_data: Whether to restore data
            restore_config: Whether to restore configuration
            restore_code: Whether to restore code

        Returns:
            True if restore was successful
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            logger.info(f"Restoring from backup: {backup_path}")

            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract backup
                await self._extract_archive(backup_file, temp_path)

                # Find backup root directory
                backup_root = None
                for item in temp_path.iterdir():
                    if item.is_dir():
                        backup_root = item
                        break

                if backup_root is None:
                    logger.error("Invalid backup structure")
                    return False

                # Read manifest
                manifest_file = backup_root / "manifest.json"
                if manifest_file.exists():
                    with open(manifest_file) as f:
                        manifest = json.load(f)
                    logger.info(f"Restoring backup from {manifest.get('timestamp')}")

                # Stop services before restore
                await self._stop_services()

                try:
                    # Restore components
                    if restore_config:
                        await self._restore_configuration(backup_root / "config")

                    if restore_data:
                        await self._restore_data(backup_root / "data")

                    if restore_code:
                        await self._restore_code(backup_root / "code")

                    logger.info("Restore completed successfully")
                    return True

                finally:
                    # Always try to restart services
                    await self._start_services()

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    async def list_backups(self) -> List[Dict[str, any]]:
        """List available backups.

        Returns:
            List of backup information
        """
        try:
            backups = []

            for backup_file in self.backup_dir.glob("*.tar.gz"):
                # Get file stats
                stat = backup_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                created = datetime.datetime.fromtimestamp(stat.st_mtime)

                # Try to read manifest for more info
                manifest_info = {}
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)

                        # Extract just the manifest
                        with tarfile.open(backup_file, 'r:gz') as tar:
                            for member in tar.getmembers():
                                if member.name.endswith('manifest.json'):
                                    tar.extract(member, temp_path)
                                    manifest_path = temp_path / member.name
                                    with open(manifest_path) as f:
                                        manifest_info = json.load(f)
                                    break

                except Exception:
                    # Ignore manifest read errors
                    pass

                backup_info = {
                    "name": backup_file.stem.replace('.tar', ''),
                    "path": str(backup_file),
                    "size_mb": round(size_mb, 2),
                    "created": created.isoformat(),
                    "type": manifest_info.get("type", "unknown"),
                    "version": manifest_info.get("version", "unknown")
                }
                backups.append(backup_info)

            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created"], reverse=True)

            return backups

        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []

    async def cleanup_old_backups(self, keep_count: int = 10) -> None:
        """Clean up old backup files.

        Args:
            keep_count: Number of backups to keep
        """
        try:
            logger.info(f"Cleaning up old backups (keeping {keep_count})")

            backup_files = list(self.backup_dir.glob("*.tar.gz"))
            backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove old backups
            for old_backup in backup_files[keep_count:]:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup.name}")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    # Private helper methods

    async def _backup_code(self, target_dir: Path) -> None:
        """Backup application code."""
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy backend code
        backend_src = self.system_dir / "backend"
        if backend_src.exists():
            shutil.copytree(backend_src, target_dir / "backend")

        # Copy deployment scripts
        deployment_src = self.system_dir / "deployment"
        if deployment_src.exists():
            shutil.copytree(deployment_src, target_dir / "deployment")

        logger.debug("Code backup completed")

    async def _backup_configuration(self, target_dir: Path) -> None:
        """Backup configuration files."""
        target_dir.mkdir(parents=True, exist_ok=True)

        config_src = self.system_dir / "config"
        if config_src.exists():
            shutil.copytree(config_src, target_dir)

        logger.debug("Configuration backup completed")

    async def _backup_data(self, target_dir: Path) -> None:
        """Backup application data."""
        target_dir.mkdir(parents=True, exist_ok=True)

        data_src = self.system_dir / "data"
        if data_src.exists():
            shutil.copytree(data_src, target_dir)

        logger.debug("Data backup completed")

    async def _backup_logs(self, target_dir: Path) -> None:
        """Backup recent log files."""
        target_dir.mkdir(parents=True, exist_ok=True)

        logs_src = self.system_dir / "logs"
        if logs_src.exists():
            # Only backup recent logs (last 7 days)
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=7)

            for log_file in logs_src.glob("*.log*"):
                if log_file.stat().st_mtime > cutoff_time.timestamp():
                    shutil.copy2(log_file, target_dir)

        logger.debug("Logs backup completed")

    async def _backup_system_info(self, target_dir: Path) -> None:
        """Backup system information."""
        target_dir.mkdir(parents=True, exist_ok=True)

        # Get system information
        system_info = {}

        try:
            # Systemd service status
            result = subprocess.run([
                "systemctl", "show", "billiards-trainer", "--no-page"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                system_info["service_status"] = result.stdout

        except Exception:
            pass

        # Save system info
        with open(target_dir / "system_info.json", 'w') as f:
            json.dump(system_info, f, indent=2)

        logger.debug("System info backup completed")

    async def _create_manifest(self) -> Dict[str, any]:
        """Create backup manifest."""
        return {
            "type": "full",
            "timestamp": datetime.datetime.now().isoformat(),
            "version": await self._get_system_version(),
            "system": {
                "platform": os.uname().sysname,
                "hostname": os.uname().nodename,
            }
        }

    async def _get_system_version(self) -> str:
        """Get system version."""
        try:
            version_file = self.system_dir / "VERSION"
            if version_file.exists():
                return version_file.read_text().strip()
        except Exception:
            pass

        return "unknown"

    async def _create_archive(self, source_dir: Path, archive_path: Path) -> None:
        """Create compressed archive."""
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(source_dir, arcname=source_dir.name)

    async def _extract_archive(self, archive_path: Path, target_dir: Path) -> None:
        """Extract compressed archive."""
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(target_dir)

    async def _stop_services(self) -> None:
        """Stop system services."""
        try:
            subprocess.run([
                "sudo", "systemctl", "stop", "billiards-trainer"
            ], check=True)
            logger.info("Services stopped")
        except Exception as e:
            logger.warning(f"Failed to stop services: {e}")

    async def _start_services(self) -> None:
        """Start system services."""
        try:
            subprocess.run([
                "sudo", "systemctl", "start", "billiards-trainer"
            ], check=True)
            logger.info("Services started")
        except Exception as e:
            logger.warning(f"Failed to start services: {e}")

    async def _restore_configuration(self, source_dir: Path) -> None:
        """Restore configuration files."""
        if not source_dir.exists():
            return

        target_dir = self.system_dir / "config"
        if target_dir.exists():
            shutil.rmtree(target_dir)

        shutil.copytree(source_dir, target_dir)
        logger.info("Configuration restored")

    async def _restore_data(self, source_dir: Path) -> None:
        """Restore data files."""
        if not source_dir.exists():
            return

        target_dir = self.system_dir / "data"
        if target_dir.exists():
            # Create backup of current data
            backup_name = f"data_backup_before_restore_{int(datetime.datetime.now().timestamp())}"
            shutil.move(target_dir, target_dir.parent / backup_name)

        shutil.copytree(source_dir, target_dir)
        logger.info("Data restored")

    async def _restore_code(self, source_dir: Path) -> None:
        """Restore application code."""
        if not source_dir.exists():
            return

        # Restore backend
        backend_target = self.system_dir / "backend"
        if backend_target.exists():
            shutil.rmtree(backend_target)

        backend_source = source_dir / "backend"
        if backend_source.exists():
            shutil.copytree(backend_source, backend_target)

        # Restore deployment
        deployment_target = self.system_dir / "deployment"
        if deployment_target.exists():
            shutil.rmtree(deployment_target)

        deployment_source = source_dir / "deployment"
        if deployment_source.exists():
            shutil.copytree(deployment_source, deployment_target)

        logger.info("Code restored")


async def main():
    """Main backup script function."""
    parser = argparse.ArgumentParser(description="Billiards Trainer Backup Manager")

    parser.add_argument(
        "action",
        choices=["backup", "restore", "list", "cleanup"],
        help="Action to perform"
    )

    parser.add_argument(
        "--system-dir",
        default="/opt/billiards-trainer",
        help="System directory to backup/restore"
    )

    parser.add_argument(
        "--backup-dir",
        default="/opt/billiards-trainer/backups",
        help="Backup storage directory"
    )

    parser.add_argument(
        "--name",
        help="Backup name (for backup action)"
    )

    parser.add_argument(
        "--backup-file",
        help="Backup file to restore (for restore action)"
    )

    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Backup/restore data only"
    )

    parser.add_argument(
        "--keep-count",
        type=int,
        default=10,
        help="Number of backups to keep (for cleanup action)"
    )

    args = parser.parse_args()

    # Create backup manager
    backup_manager = BackupManager(args.system_dir, args.backup_dir)

    try:
        if args.action == "backup":
            if args.data_only:
                path = await backup_manager.create_data_backup(args.name)
            else:
                path = await backup_manager.create_full_backup(args.name)

            print(f"Backup created: {path}")

        elif args.action == "restore":
            if not args.backup_file:
                logger.error("--backup-file is required for restore action")
                sys.exit(1)

            success = await backup_manager.restore_backup(
                args.backup_file,
                restore_data=True,
                restore_config=True,
                restore_code=not args.data_only
            )

            if success:
                print("Restore completed successfully")
            else:
                print("Restore failed")
                sys.exit(1)

        elif args.action == "list":
            backups = await backup_manager.list_backups()

            print(f"{'Name':<30} {'Type':<10} {'Size (MB)':<10} {'Created':<20}")
            print("-" * 70)

            for backup in backups:
                print(f"{backup['name']:<30} {backup['type']:<10} {backup['size_mb']:<10} {backup['created'][:19]}")

        elif args.action == "cleanup":
            await backup_manager.cleanup_old_backups(args.keep_count)
            print(f"Cleanup completed (kept {args.keep_count} backups)")

    except Exception as e:
        logger.error(f"Action failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    asyncio.run(main())
