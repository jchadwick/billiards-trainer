#!/usr/bin/env bash
set -euo pipefail

image=${DOCKER_IMAGE:-billiards-trainer:dev}
startup_timeout=${DOCKER_SMOKE_TIMEOUT_SECONDS:-120}
container_name="billiards-runtime-smoke-${$}-${RANDOM}"
temporary_directory=$(mktemp -d)

case "$startup_timeout" in
    ''|*[!0-9]*)
        echo "DOCKER_SMOKE_TIMEOUT_SECONDS must be a positive integer" >&2
        exit 2
        ;;
esac
if (( startup_timeout < 1 )); then
    echo "DOCKER_SMOKE_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi

cleanup() {
    docker rm --force "$container_name" >/dev/null 2>&1 || true
    docker volume rm "$container_name" >/dev/null 2>&1 || true
    rm -rf "$temporary_directory"
}
trap cleanup EXIT

docker run --detach --rm \
    --name "$container_name" \
    --publish 127.0.0.1::8000 \
    --env START_LIVE=true \
    --mount "type=volume,source=$container_name,target=/app/backend/calibration_data" \
    "$image" >/dev/null

published_address=$(docker port "$container_name" 8000/tcp | head -n 1)
host_port=${published_address##*:}
if [[ ! "$host_port" =~ ^[0-9]+$ ]]; then
    echo "Docker did not publish a valid local gateway port: $published_address" >&2
    docker logs "$container_name" >&2 || true
    exit 1
fi

base_url="http://127.0.0.1:${host_port}"
deadline=$((SECONDS + startup_timeout))
health_status=starting
while (( SECONDS < deadline )); do
    health_status=$(docker inspect --format '{{.State.Health.Status}}' "$container_name")
    if [[ "$health_status" == healthy ]]; then
        break
    fi
    container_status=$(docker inspect --format '{{.State.Status}}' "$container_name")
    if [[ "$container_status" != running ]]; then
        echo "Container exited before becoming healthy (state: $container_status)" >&2
        docker logs "$container_name" >&2 || true
        exit 1
    fi
    sleep 1
done

if [[ "$health_status" != healthy ]]; then
    echo "Container did not become healthy within ${startup_timeout}s (health: $health_status)" >&2
    docker logs "$container_name" >&2 || true
    exit 1
fi

curl --fail --silent --show-error \
    "$base_url/api/v1/health/live" >"$temporary_directory/health.json"
curl --fail --silent --show-error "$base_url/api/v1/setup" >"$temporary_directory/setup.json"
if ! grep -Eq '"state"[[:space:]]*:[[:space:]]*"idle"' "$temporary_directory/setup.json"; then
    echo "Fresh container did not wait for camera selection" >&2
    cat "$temporary_directory/setup.json" >&2
    exit 1
fi
curl --fail --silent --show-error "$base_url/live" >"$temporary_directory/live.html"
if ! grep -q 'id="app"' "$temporary_directory/live.html"; then
    echo "The /live route did not serve the browser application entry point" >&2
    exit 1
fi

echo "Docker runtime smoke passed: healthy gateway, idle setup, and /live served by $image."
