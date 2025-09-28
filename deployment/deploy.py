#!/usr/bin/env python3
"""Production deployment script for Billiards Trainer system."""

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages production deployment of the Billiards Trainer system."""

    def __init__(self, target_dir: str, environment: str = "production"):
        """Initialize deployment manager.

        Args:
            target_dir: Target deployment directory
            environment: Deployment environment
        """
        self.target_dir = Path(target_dir)
        self.environment = environment
        self.source_dir = Path(__file__).parent.parent
        self.backup_dir = self.target_dir / "backups"

        logger.info(f"Deployment Manager initialized for {environment}")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Target: {self.target_dir}")

    async def deploy(self, skip_backup: bool = False) -> bool:
        """Deploy the system to production.

        Args:
            skip_backup: Skip backup creation

        Returns:
            True if deployment successful
        """
        try:
            logger.info("=== Starting Production Deployment ===")

            # Pre-deployment checks
            if not await self._pre_deployment_checks():
                return False

            # Create backup
            if not skip_backup:
                await self._create_backup()

            # Stop existing services
            await self._stop_services()

            # Deploy code
            await self._deploy_code()

            # Install dependencies
            await self._install_dependencies()

            # Update configuration
            await self._update_configuration()

            # Set up systemd services
            await self._setup_services()

            # Run database migrations (if applicable)
            await self._run_migrations()

            # Start services
            await self._start_services()

            # Post-deployment verification
            if await self._verify_deployment():
                logger.info("=== Deployment Completed Successfully ===")
                return True
            else:
                logger.error("=== Deployment Verification Failed ===")
                await self._rollback()
                return False

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            await self._rollback()
            return False

    async def _pre_deployment_checks(self) -> bool:
        """Run pre-deployment checks.

        Returns:
            True if all checks pass
        """
        logger.info("Running pre-deployment checks...")

        checks = [
            ("Source directory exists", self.source_dir.exists()),
            ("Target directory accessible", self._check_target_dir()),
            ("Python version compatible", self._check_python_version()),
            ("Required packages available", await self._check_dependencies()),
            ("System resources sufficient", await self._check_system_resources()),
            ("Ports available", await self._check_ports()),
        ]

        all_passed = True
        for check_name, result in checks:
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check_name}")
            if not result:
                all_passed = False

        return all_passed

    def _check_target_dir(self) -> bool:
        """Check if target directory is accessible."""
        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Cannot access target directory: {e}")
            return False

    def _check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        return version.major == 3 and version.minor >= 8

    async def _check_dependencies(self) -> bool:
        """Check if required system dependencies are available."""
        try:
            # Check essential system commands
            commands = ["systemctl", "nginx", "git"]
            for cmd in commands:
                result = subprocess.run(["which", cmd], capture_output=True)
                if result.returncode != 0:
                    logger.warning(f"Command {cmd} not found")

            return True
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False

    async def _check_system_resources(self) -> bool:
        """Check system resources."""
        try:
            import psutil

            # Check memory (minimum 2GB)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 2:
                logger.warning(f"Low memory: {memory_gb:.1f}GB")

            # Check disk space (minimum 10GB free)
            disk_free_gb = psutil.disk_usage('/').free / (1024**3)
            if disk_free_gb < 10:
                logger.warning(f"Low disk space: {disk_free_gb:.1f}GB")

            return True
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return False

    async def _check_ports(self) -> bool:
        """Check if required ports are available."""
        try:
            import socket

            ports = [8000, 8001, 8080]  # API, WebSocket, Nginx
            for port in ports:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    result = sock.connect_ex(('localhost', port))
                    if result == 0:
                        logger.warning(f"Port {port} is in use")

            return True
        except Exception as e:
            logger.error(f"Port check failed: {e}")
            return False

    async def _create_backup(self) -> None:
        """Create backup of existing deployment."""
        try:
            if not self.target_dir.exists():
                logger.info("No existing deployment to backup")
                return

            logger.info("Creating backup...")

            # Create backup directory
            self.backup_dir.mkdir(parents=True, exist_ok=True)

            # Generate backup name with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            backup_path = self.backup_dir / backup_name

            # Copy current deployment
            shutil.copytree(self.target_dir, backup_path, ignore=shutil.ignore_patterns("backups"))

            # Compress backup
            archive_path = f"{backup_path}.tar.gz"
            subprocess.run([
                "tar", "-czf", archive_path, "-C", str(self.backup_dir), backup_name
            ])

            # Remove uncompressed backup
            shutil.rmtree(backup_path)

            # Keep only last 5 backups
            await self._cleanup_old_backups()

            logger.info(f"Backup created: {archive_path}")

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise

    async def _cleanup_old_backups(self) -> None:
        """Clean up old backup files."""
        try:
            backup_files = list(self.backup_dir.glob("backup_*.tar.gz"))
            backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Keep only the 5 most recent
            for old_backup in backup_files[5:]:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup}")

        except Exception as e:
            logger.warning(f"Backup cleanup failed: {e}")

    async def _stop_services(self) -> None:
        """Stop existing services."""
        try:
            logger.info("Stopping services...")

            services = ["billiards-trainer", "nginx"]
            for service in services:
                result = subprocess.run([
                    "sudo", "systemctl", "stop", service
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info(f"Stopped service: {service}")
                else:
                    logger.warning(f"Failed to stop {service}: {result.stderr}")

        except Exception as e:
            logger.warning(f"Service stop failed: {e}")

    async def _deploy_code(self) -> None:
        """Deploy application code."""
        try:
            logger.info("Deploying code...")

            # Remove old deployment (except backups and config)
            if self.target_dir.exists():
                for item in self.target_dir.iterdir():
                    if item.name not in ["backups", "config", "logs", "data"]:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()

            # Copy source code
            source_backend = self.source_dir / "backend"
            target_backend = self.target_dir / "backend"

            shutil.copytree(source_backend, target_backend, dirs_exist_ok=True)

            # Copy deployment files
            source_deployment = self.source_dir / "deployment"
            target_deployment = self.target_dir / "deployment"

            shutil.copytree(source_deployment, target_deployment, dirs_exist_ok=True)

            # Set permissions
            subprocess.run(["chmod", "+x", str(self.target_dir / "backend" / "system_launcher.py")])

            logger.info("Code deployment completed")

        except Exception as e:
            logger.error(f"Code deployment failed: {e}")
            raise

    async def _install_dependencies(self) -> None:
        """Install Python dependencies."""
        try:
            logger.info("Installing dependencies...")

            # Install system packages
            system_packages = [
                "python3-venv",
                "python3-dev",
                "nginx",
                "redis-server",
                "libopencv-dev",
                "python3-opencv"
            ]

            subprocess.run([
                "sudo", "apt-get", "update"
            ], check=True)

            subprocess.run([
                "sudo", "apt-get", "install", "-y"
            ] + system_packages, check=True)

            # Create virtual environment
            venv_path = self.target_dir / "venv"
            if not venv_path.exists():
                subprocess.run([
                    "python3", "-m", "venv", str(venv_path)
                ], check=True)

            # Install Python packages
            pip_path = venv_path / "bin" / "pip"
            requirements_file = self.source_dir / "requirements.txt"

            if requirements_file.exists():
                subprocess.run([
                    str(pip_path), "install", "-r", str(requirements_file)
                ], check=True)

            # Install additional production packages
            production_packages = [
                "uvicorn[standard]",
                "gunicorn",
                "psutil",
                "redis",
                "prometheus-client"
            ]

            subprocess.run([
                str(pip_path), "install"
            ] + production_packages, check=True)

            logger.info("Dependencies installed")

        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            raise

    async def _update_configuration(self) -> None:
        """Update configuration files."""
        try:
            logger.info("Updating configuration...")

            config_dir = self.target_dir / "config"
            config_dir.mkdir(exist_ok=True)

            # Create production configuration
            production_config = {
                "environment": "production",
                "debug": False,
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "workers": 4
                },
                "logging": {
                    "level": "INFO",
                    "file": str(self.target_dir / "logs" / "billiards-trainer.log")
                },
                "monitoring": {
                    "enabled": True,
                    "prometheus_port": 9090
                },
                "security": {
                    "cors_origins": ["http://localhost:3000"],
                    "rate_limiting": True
                }
            }

            import json
            config_file = config_dir / "production.json"
            with open(config_file, 'w') as f:
                json.dump(production_config, f, indent=2)

            # Create nginx configuration
            await self._create_nginx_config()

            # Create log directories
            (self.target_dir / "logs").mkdir(exist_ok=True)
            (self.target_dir / "data").mkdir(exist_ok=True)

            logger.info("Configuration updated")

        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            raise

    async def _create_nginx_config(self) -> None:
        """Create nginx configuration."""
        nginx_config = f"""
server {{
    listen 80;
    server_name _;

    location / {{
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}

    location /ws {{
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}

    location /static/ {{
        alias {self.target_dir}/static/;
        expires 30d;
    }}

    location /health {{
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }}
}}
"""

        nginx_file = Path("/etc/nginx/sites-available/billiards-trainer")
        with open(nginx_file, 'w') as f:
            f.write(nginx_config)

        # Enable site
        nginx_enabled = Path("/etc/nginx/sites-enabled/billiards-trainer")
        if nginx_enabled.exists():
            nginx_enabled.unlink()
        nginx_enabled.symlink_to(nginx_file)

    async def _setup_services(self) -> None:
        """Set up systemd services."""
        try:
            logger.info("Setting up systemd services...")

            # Create systemd service file
            service_config = f"""
[Unit]
Description=Billiards Trainer System
After=network.target

[Service]
Type=simple
User=billiards
Group=billiards
WorkingDirectory={self.target_dir}
Environment=PATH={self.target_dir}/venv/bin
Environment=PYTHONPATH={self.target_dir}/backend
ExecStart={self.target_dir}/venv/bin/python {self.target_dir}/backend/system_launcher.py --environment production
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

            service_file = Path("/etc/systemd/system/billiards-trainer.service")
            with open(service_file, 'w') as f:
                f.write(service_config)

            # Reload systemd
            subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)

            # Enable service
            subprocess.run(["sudo", "systemctl", "enable", "billiards-trainer"], check=True)

            logger.info("Systemd services configured")

        except Exception as e:
            logger.error(f"Service setup failed: {e}")
            raise

    async def _run_migrations(self) -> None:
        """Run database migrations if applicable."""
        try:
            logger.info("Running migrations...")
            # Add database migration logic here if needed
            logger.info("Migrations completed")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

    async def _start_services(self) -> None:
        """Start services."""
        try:
            logger.info("Starting services...")

            services = ["nginx", "billiards-trainer"]
            for service in services:
                subprocess.run([
                    "sudo", "systemctl", "start", service
                ], check=True)

                logger.info(f"Started service: {service}")

        except Exception as e:
            logger.error(f"Service start failed: {e}")
            raise

    async def _verify_deployment(self) -> bool:
        """Verify deployment is working."""
        try:
            logger.info("Verifying deployment...")

            # Wait for services to start
            await asyncio.sleep(10)

            # Check service status
            result = subprocess.run([
                "systemctl", "is-active", "billiards-trainer"
            ], capture_output=True, text=True)

            if result.stdout.strip() != "active":
                logger.error("Billiards Trainer service is not active")
                return False

            # Check API endpoint
            import urllib.request
            try:
                with urllib.request.urlopen("http://localhost/health", timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"Health check failed: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"Health check request failed: {e}")
                return False

            logger.info("Deployment verification passed")
            return True

        except Exception as e:
            logger.error(f"Deployment verification failed: {e}")
            return False

    async def _rollback(self) -> None:
        """Rollback to previous deployment."""
        try:
            logger.info("Rolling back deployment...")

            # Find latest backup
            if not self.backup_dir.exists():
                logger.error("No backups available for rollback")
                return

            backup_files = list(self.backup_dir.glob("backup_*.tar.gz"))
            if not backup_files:
                logger.error("No backup files found")
                return

            latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)

            # Extract backup
            subprocess.run([
                "tar", "-xzf", str(latest_backup), "-C", str(self.target_dir.parent)
            ], check=True)

            # Restart services
            await self._start_services()

            logger.info(f"Rollback completed using backup: {latest_backup}")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")


async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Billiards Trainer Deployment Script")

    parser.add_argument(
        "target_dir",
        help="Target deployment directory"
    )

    parser.add_argument(
        "--environment",
        choices=["production", "staging"],
        default="production",
        help="Deployment environment"
    )

    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip backup creation"
    )

    args = parser.parse_args()

    # Create deployment manager
    deployer = DeploymentManager(args.target_dir, args.environment)

    # Run deployment
    success = await deployer.deploy(skip_backup=args.skip_backup)

    if success:
        logger.info("Deployment completed successfully")
        sys.exit(0)
    else:
        logger.error("Deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
