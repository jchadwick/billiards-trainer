#!/usr/bin/env bash
# Start the packaged Rust gateway and its supervised Rust live detector.
# This script intentionally does not install dependencies or invoke Python.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${BILLIARDS_BIN_DIR:-$SCRIPT_DIR/bin}"
SERVER="${BILLIARDS_SERVER_EXECUTABLE:-$BIN_DIR/billiards-server}"
LIVE="${BILLIARDS_LIVE_EXECUTABLE:-$BIN_DIR/billiards-perception-live}"
CAPTURE_HELPER="${BILLIARDS_CAPTURE_HELPER_EXECUTABLE:-$BIN_DIR/billiards-perception-gstreamer-probe}"
CONFIG="${BILLIARDS_CONFIG:-$SCRIPT_DIR/config.json}"
WEB_ROOT="${BILLIARDS_WEB_ROOT:-$SCRIPT_DIR/frontend}"
PROJECT_ROOT="${BILLIARDS_PROJECT_ROOT:-$SCRIPT_DIR}"
RUN_DIR="${BILLIARDS_RUN_DIR:-$SCRIPT_DIR/run}"
PID_FILE="${BILLIARDS_PID_FILE:-$RUN_DIR/billiards-server.pid}"
HTTP_BIND="${BILLIARDS_HTTP_BIND:-0.0.0.0}"
HTTP_PORT="${BILLIARDS_HTTP_PORT:-8000}"
SCENE_BIND="${BILLIARDS_SCENE_BIND:-127.0.0.1}"
SCENE_PORT="${BILLIARDS_SCENE_PORT:-49321}"
START_LIVE="${BILLIARDS_START_LIVE:-true}"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_executable() {
    local executable="$1"
    local label="$2"
    [[ -f "$executable" && -x "$executable" ]] || fail "$label is missing or not executable: $executable. Build or install the Rust distribution first."
}

require_file() {
    local file_path="$1"
    local label="$2"
    [[ -f "$file_path" && -r "$file_path" && -s "$file_path" ]] || fail "$label is missing, unreadable, or empty: $file_path"
}

[[ -f "$CONFIG" && -r "$CONFIG" && -s "$CONFIG" ]] || fail "configuration file is missing, unreadable, or empty: $CONFIG"
[[ -d "$PROJECT_ROOT" ]] || fail "project root does not exist: $PROJECT_ROOT"
require_executable "$SERVER" "Rust server binary"
if [[ "$START_LIVE" == "true" || "$START_LIVE" == "1" || "$START_LIVE" == "yes" ]]; then
    require_executable "$LIVE" "Rust live detector binary"
    require_executable "$CAPTURE_HELPER" "Rust GStreamer capture helper"
    if [[ "$CONFIG" == "$SCRIPT_DIR/config.json" ]]; then
        # The distribution builder records the selected provider separately so
        # startup can validate the right model bundle without adding a JSON
        # parser dependency to the target device.
        inference_provider="onnx"
        if [[ -f "$SCRIPT_DIR/.billiards-inference-provider" ]]; then
            inference_provider="$(cat -- "$SCRIPT_DIR/.billiards-inference-provider")"
        fi
        case "$inference_provider" in
            openvino)
                require_file "$PROJECT_ROOT/models/billiards-yolov8n.onnx" "packaged ONNX model"
                require_file "$PROJECT_ROOT/models/billiards-yolov8n.pt" "packaged source checkpoint"
                require_file "$PROJECT_ROOT/models/billiards-yolov8n.onnx.manifest.json" "packaged ONNX export manifest"
                ;;
            *)
                fail "unsupported packaged inference provider '$inference_provider'"
                ;;
        esac
    fi
elif [[ "$START_LIVE" != "false" && "$START_LIVE" != "0" && "$START_LIVE" != "no" ]]; then
    fail "BILLIARDS_START_LIVE must be true or false (received: $START_LIVE)"
fi
if [[ ! -d "$WEB_ROOT" || ! -f "$WEB_ROOT/index.html" ]]; then
    fail "built web client is missing from $WEB_ROOT (expected index.html); build frontend/web before starting the server"
fi

mkdir -p "$(dirname -- "$PID_FILE")" || fail "could not create PID-file directory: $(dirname -- "$PID_FILE")"
if [[ -e "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        existing_command="$(ps -p "$existing_pid" -o command= 2>/dev/null || true)"
        case "$existing_command" in
            *"$SERVER"*) fail "this distribution already has a server running (PID $existing_pid)" ;;
            *) fail "PID file $PID_FILE refers to live process $existing_pid, not this Rust server; remove the stale PID file only after verifying that process" ;;
        esac
    fi
    rm -f -- "$PID_FILE"
fi

args=(
    --http-bind "$HTTP_BIND"
    --http-port "$HTTP_PORT"
    --scene-bind "$SCENE_BIND"
    --scene-port "$SCENE_PORT"
    --config "$CONFIG"
    --project-root "$PROJECT_ROOT"
    --web-root "$WEB_ROOT"
)
if [[ "$START_LIVE" == "true" || "$START_LIVE" == "1" || "$START_LIVE" == "yes" ]]; then
    args+=(
        --start-live
        --live-executable "$LIVE"
        --capture-helper-executable "$CAPTURE_HELPER"
    )
fi

printf 'Starting Rust Billiards Trainer server at http://%s:%s\n' "$HTTP_BIND" "$HTTP_PORT"
printf 'Serving web client from %s; live detector: %s\n' "$WEB_ROOT" "$START_LIVE"
"$SERVER" "${args[@]}" &
server_pid=$!
printf '%s\n' "$server_pid" > "$PID_FILE"

cleanup() {
    local status="$1"
    trap - INT TERM EXIT
    if kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$server_pid" ]]; then
        rm -f -- "$PID_FILE"
    fi
    exit "$status"
}

trap 'cleanup 130' INT
trap 'cleanup 143' TERM
trap 'cleanup $?' EXIT

if wait "$server_pid"; then
    status=0
else
    status=$?
fi
trap - EXIT
if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$server_pid" ]]; then
    rm -f -- "$PID_FILE"
fi
exit "$status"
