#!/bin/bash

# Example deployment script for Billiards Trainer
# ==============================================
# This script demonstrates how to deploy the system using the built Docker images
# Customize this script according to your deployment environment

set -e

# Configuration
REGISTRY="${REGISTRY:-ghcr.io}"
REPOSITORY="${REPOSITORY:-jchadwick/billiards-trainer}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENVIRONMENT="${ENVIRONMENT:-staging}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

echo "🚀 Starting deployment for environment: $ENVIRONMENT"
echo "Registry: $REGISTRY"
echo "Repository: $REPOSITORY"
echo "Image Tag: $IMAGE_TAG"
echo "Compose File: $COMPOSE_FILE"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate requirements
if ! command_exists docker; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi

# Choose docker-compose command
COMPOSE_CMD="docker-compose"
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
fi

# Pull latest images
echo "📥 Pulling latest images..."
docker pull "$REGISTRY/$REPOSITORY/backend:$IMAGE_TAG"

# Create environment file for deployment
echo "📝 Creating deployment environment file..."
cat > .env.deploy << EOF
# Deployment environment: $ENVIRONMENT
BUILD_VERSION=$IMAGE_TAG
ENVIRONMENT=$ENVIRONMENT

# Override image registry
BACKEND_IMAGE=$REGISTRY/$REPOSITORY/backend:$IMAGE_TAG

# Add your environment-specific variables here
# Example:
# API_PORT=8000
# WS_PORT=8001
# REDIS_PORT=6379
EOF

# Deployment strategies based on environment
case "$ENVIRONMENT" in
    "staging")
        echo "🔧 Deploying to staging environment..."

        # Stop existing services
        $COMPOSE_CMD --env-file .env.deploy down

        # Start services
        $COMPOSE_CMD --env-file .env.deploy up -d backend redis

        # Wait for health check
        echo "⏳ Waiting for services to be healthy..."
        sleep 10

        # Verify deployment
        if curl -f "http://localhost:8000/api/v1/health"; then
            echo "✅ Staging deployment successful!"
        else
            echo "❌ Health check failed"
            $COMPOSE_CMD --env-file .env.deploy logs backend
            exit 1
        fi
        ;;

    "production")
        echo "🔧 Deploying to production environment..."

        # Production deployment with zero-downtime (example)
        # 1. Start new services alongside old ones
        # 2. Switch traffic
        # 3. Stop old services

        echo "⚠️  Production deployment strategy should be customized for your infrastructure"
        echo "Consider using:"
        echo "- Blue-green deployment"
        echo "- Rolling updates"
        echo "- Load balancer integration"
        echo "- Database migrations"

        # Basic production deployment (customize as needed)
        $COMPOSE_CMD --env-file .env.deploy up -d --remove-orphans

        echo "✅ Production deployment initiated"
        ;;

    *)
        echo "❌ Unknown environment: $ENVIRONMENT"
        echo "Supported environments: staging, production"
        exit 1
        ;;
esac

# Cleanup
echo "🧹 Cleaning up..."
rm -f .env.deploy

# Display deployment information
echo ""
echo "🎉 Deployment Summary"
echo "===================="
echo "Environment: $ENVIRONMENT"
echo "Image: $REGISTRY/$REPOSITORY/backend:$IMAGE_TAG"
echo "Services: Backend API, Redis Cache"
echo ""
echo "Next steps:"
echo "- Monitor application logs: $COMPOSE_CMD logs -f backend"
echo "- Check service status: $COMPOSE_CMD ps"
echo "- Verify health: curl http://localhost:8000/api/v1/health"
echo ""
echo "For production deployments, ensure you have:"
echo "- Proper environment variables configured"
echo "- SSL/TLS certificates in place"
echo "- Monitoring and alerting setup"
echo "- Backup and recovery procedures"
