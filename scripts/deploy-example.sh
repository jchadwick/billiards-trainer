#!/usr/bin/env bash

# Build and run the Compose services locally. This script does not contact a
# registry or remote host and does not read or request authentication secrets.

set -euo pipefail

ENVIRONMENT="${ENVIRONMENT:-staging}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required but is not installed" >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is required but is not installed" >&2
    exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "Compose file not found: $COMPOSE_FILE" >&2
    exit 1
fi

COMPOSE=(docker compose --file "$COMPOSE_FILE")

echo "Building local images from $COMPOSE_FILE"
"${COMPOSE[@]}" build

case "$ENVIRONMENT" in
    staging)
        echo "Starting local backend service"
        "${COMPOSE[@]}" up -d backend
        ;;
    production)
        echo "Starting all local Compose services"
        "${COMPOSE[@]}" up -d --remove-orphans
        ;;
    *)
        echo "Unsupported environment: $ENVIRONMENT (choose staging or production)" >&2
        exit 1
        ;;
esac

echo "Waiting for the local backend health endpoint"
for attempt in {1..30}; do
    if curl --fail --silent http://localhost:8000/api/v1/health >/dev/null; then
        echo "Local service is healthy at http://localhost:8000"
        exit 0
    fi
    sleep 1
done

echo "Backend health check failed; recent logs follow:" >&2
"${COMPOSE[@]}" logs --tail=100 backend >&2
exit 1
