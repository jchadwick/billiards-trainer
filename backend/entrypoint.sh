#!/bin/sh
set -eu

# Keep the container's process tree owned by the Rust gateway. The gateway
# supervises billiards-perception-live and forwards shutdown to that child.
CONFIG_FILE=${CONFIG_FILE:-/app/config/config.json}
PROJECT_ROOT=${PROJECT_ROOT:-/app}
WEB_ROOT=${WEB_ROOT:-/app/web}
HTTP_BIND=${HTTP_BIND:-0.0.0.0}
HTTP_PORT=${HTTP_PORT:-8000}
SCENE_BIND=${SCENE_BIND:-127.0.0.1}
SCENE_PORT=${SCENE_PORT:-49321}
LIVE_EXECUTABLE=${LIVE_EXECUTABLE:-/app/bin/billiards-perception-live}
CAPTURE_HELPER_EXECUTABLE=${CAPTURE_HELPER_EXECUTABLE:-/app/bin/billiards-perception-gstreamer-probe}
START_LIVE=${START_LIVE:-true}

case "$START_LIVE" in
    true|1|yes)
        LIVE_ENABLED=true
        ;;
    false|0|no)
        LIVE_ENABLED=false
        ;;
    *)
        echo "START_LIVE must be true/false, 1/0, or yes/no (got: $START_LIVE)" >&2
        exit 2
        ;;
esac

mkdir -p "$PROJECT_ROOT" /app/config /app/data /app/logs /app/temp

if [ ! -d "$WEB_ROOT" ]; then
    echo "Rust server static web root does not exist: $WEB_ROOT" >&2
    exit 1
fi

set -- \
    --http-bind "$HTTP_BIND" \
    --http-port "$HTTP_PORT" \
    --scene-bind "$SCENE_BIND" \
    --scene-port "$SCENE_PORT" \
    --project-root "$PROJECT_ROOT" \
    --web-root "$WEB_ROOT"

if [ -f "$CONFIG_FILE" ]; then
    set -- "$@" --config "$CONFIG_FILE"
elif [ "$LIVE_ENABLED" = true ]; then
    echo "Live capture requires a mounted configuration file: $CONFIG_FILE" >&2
    exit 1
else
    echo "No configuration file found; starting gateway-only without config APIs."
fi

case "$LIVE_ENABLED" in
    true)
        if [ ! -x "$LIVE_EXECUTABLE" ]; then
            echo "Rust live detector is not executable: $LIVE_EXECUTABLE" >&2
            exit 1
        fi
        if [ ! -x "$CAPTURE_HELPER_EXECUTABLE" ]; then
            echo "Rust GStreamer helper is not executable: $CAPTURE_HELPER_EXECUTABLE" >&2
            exit 1
        fi
        set -- "$@" \
            --start-live \
            --live-executable "$LIVE_EXECUTABLE" \
            --capture-helper-executable "$CAPTURE_HELPER_EXECUTABLE"
        ;;
    false)
        ;;
esac

echo "Starting Rust billiards server on ${HTTP_BIND}:${HTTP_PORT}."
echo "Project root: ${PROJECT_ROOT}; static client: ${WEB_ROOT}; live capture: ${LIVE_ENABLED}."
exec /app/bin/billiards-server "$@"
