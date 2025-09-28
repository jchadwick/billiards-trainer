#!/usr/bin/env python3
"""Initialize authentication database."""

import logging
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config.auth_config import get_auth_config
from database.connection import db_manager, init_database

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Initialize the authentication database."""
    try:
        logger.info("🚀 Starting authentication database initialization...")

        # Get configuration
        config = get_auth_config()
        logger.info(f"📋 Using database: {config.database_url}")

        # Initialize database
        logger.info("🏗️  Creating database tables...")
        init_database()

        # Check database health
        if db_manager.health_check():
            logger.info("✅ Database health check passed")
        else:
            logger.error("❌ Database health check failed")
            return False

        logger.info("🎉 Authentication database initialization completed successfully!")

        # Print summary
        print("\n" + "=" * 60)
        print("🎯 AUTHENTICATION SYSTEM READY")
        print("=" * 60)
        print(f"📊 Database: {config.database_url}")
        print(f"🔑 JWT Algorithm: {config.jwt_algorithm}")
        print(f"⏰ Access Token Expires: {config.access_token_expire_minutes} minutes")
        print(f"🔄 Refresh Token Expires: {config.refresh_token_expire_days} days")
        print(f"🛡️  HTTPS Required: {config.require_https}")
        print(f"👥 Default Users Created: {config.create_default_users}")
        print("=" * 60)

        if config.create_default_users:
            print("\n🔐 DEFAULT USER ACCOUNTS:")
            print("  👑 admin@billiards-trainer.local / admin123!")
            print("  ⚙️  operator@billiards-trainer.local / operator123!")
            print("  👀 viewer@billiards-trainer.local / viewer123!")
            print("\n⚠️  Change these passwords in production!")

        return True

    except Exception as e:
        logger.error(f"💥 Database initialization failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
