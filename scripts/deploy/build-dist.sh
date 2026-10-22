#!/usr/bin/env bash
# Build a native Rust server distribution with the retained static web client.
# The build host and packaged target must have compatible Rust and GStreamer
# toolchains; this script does not install Python or package Python code.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONFIG_SOURCE="${BILLIARDS_CONFIG_SOURCE:-$PROJECT_ROOT/config.json}"
ONNX_SOURCE="${BILLIARDS_ONNX_MODEL:-}"
CHECKPOINT_SOURCE="${BILLIARDS_SOURCE_CHECKPOINT:-}"
DIST_DIR="${BILLIARDS_DIST_DIR:-$PROJECT_ROOT/dist}"
RUST_TARGET="${BILLIARDS_RUST_TARGET:-}"
RUST_FEATURES="${BILLIARDS_RUST_FEATURES:-camera-gstreamer-source}"
MANIFEST_SOURCE="${ONNX_SOURCE}.manifest.json"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required build tool '$1' was not found on PATH"
}

require_file() {
    [[ -f "$1" && -r "$1" && -s "$1" ]] || fail "$2 is missing, unreadable, or empty: $1"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        fail "sha256sum or shasum is required to verify the model bundle"
    fi
}

require_command cargo
require_command rustc
require_command node
require_command npm
require_command pkg-config
require_command jq
require_file "$CONFIG_SOURCE" "Rust application configuration"
[[ -n "$ONNX_SOURCE" ]] || fail "set BILLIARDS_ONNX_MODEL to the verified ONNX model to include in the distribution"
[[ -n "$CHECKPOINT_SOURCE" ]] || fail "set BILLIARDS_SOURCE_CHECKPOINT to the source .pt checkpoint matching the ONNX manifest"
require_file "$ONNX_SOURCE" "ONNX model"
require_file "$CHECKPOINT_SOURCE" "source model checkpoint"
require_file "$MANIFEST_SOURCE" "ONNX export manifest"

# Fail before building if the selected config cannot drive the Rust live path.
# In particular, the Rust camera path requires either an explicit stable
# GStreamer discovery selector or a remote HTTP-MJPEG source; integer camera
# indexes remain unsupported until the physical discovery adapter qualifies it.
jq -e '
  (.vision.camera.video_file_path == null or .vision.camera.video_file_path == "") and
  (
    ((.perception.capture.device_selector // null) | type == "string" and startswith("gstreamer-device-v1:")) or
    ((.vision.camera.device_id // null) | type == "string" and (startswith("http://") or startswith("https://")))
  )
' "$CONFIG_SOURCE" >/dev/null || fail "configuration must clear vision.camera.video_file_path and select a stable GStreamer camera or an HTTP(S) MJPEG URL; update $CONFIG_SOURCE first"

jq -e '
  .manifest_schema_version == 1 and
  (.exported_model.sha256 | type == "string" and test("^[0-9a-fA-F]{64}$")) and
  (.source_model.sha256 | type == "string" and test("^[0-9a-fA-F]{64}$"))
' "$MANIFEST_SOURCE" >/dev/null || fail "ONNX export manifest has an unsupported or incomplete schema: $MANIFEST_SOURCE"
expected_onnx_sha="$(jq -er '.exported_model.sha256 | ascii_downcase' "$MANIFEST_SOURCE")"
expected_checkpoint_sha="$(jq -er '.source_model.sha256 | ascii_downcase' "$MANIFEST_SOURCE")"
actual_onnx_sha="$(sha256_file "$ONNX_SOURCE")"
actual_checkpoint_sha="$(sha256_file "$CHECKPOINT_SOURCE")"
[[ "$actual_onnx_sha" == "$expected_onnx_sha" ]] || fail "ONNX file digest does not match its export manifest: $ONNX_SOURCE"
[[ "$actual_checkpoint_sha" == "$expected_checkpoint_sha" ]] || fail "source checkpoint digest does not match its ONNX export manifest: $CHECKPOINT_SOURCE"

pkg-config --exists gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0 || fail "GStreamer development libraries are required for camera-gstreamer-source (gstreamer-1.0, gstreamer-app-1.0, and gstreamer-video-1.0)"

if [[ "$DIST_DIR" != /* ]]; then
    DIST_DIR="$PROJECT_ROOT/$DIST_DIR"
fi
DIST_PARENT="$(dirname -- "$DIST_DIR")"
DIST_NAME="$(basename -- "$DIST_DIR")"
[[ -n "$DIST_NAME" && "$DIST_NAME" != "." && "$DIST_NAME" != ".." && "$DIST_DIR" != "/" && "$DIST_DIR" != "$PROJECT_ROOT" ]] || fail "refusing unsafe distribution output path: $DIST_DIR"
mkdir -p "$DIST_PARENT"
STAGING="$(mktemp -d "$DIST_PARENT/.${DIST_NAME}.build.XXXXXX")"
cleanup() {
    rm -rf -- "$STAGING"
}
trap cleanup EXIT INT TERM

printf '%s\n' 'Building retained HTML/JS/CSS client...'
(
    cd "$PROJECT_ROOT/frontend/web"
    npm ci
    npm run build
)
[[ -f "$PROJECT_ROOT/backend/static/index.html" ]] || fail "frontend build completed without backend/static/index.html"

printf '%s\n' 'Building Rust server and live capture binaries...'
target_args=()
if [[ -n "$RUST_TARGET" ]]; then
    target_args=(--target "$RUST_TARGET")
fi
(
    cd "$PROJECT_ROOT"
    cargo build \
        --manifest-path "$PROJECT_ROOT/perception/Cargo.toml" \
        --release \
        --features "$RUST_FEATURES" \
        "${target_args[@]}" \
        --bin billiards-server \
        --bin billiards-perception-live \
        --bin billiards-perception-gstreamer-probe
)

if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    CARGO_OUTPUT_DIR="$CARGO_TARGET_DIR"
    [[ "$CARGO_OUTPUT_DIR" == /* ]] || CARGO_OUTPUT_DIR="$PROJECT_ROOT/$CARGO_OUTPUT_DIR"
else
    CARGO_OUTPUT_DIR="$PROJECT_ROOT/perception/target"
fi
if [[ -n "$RUST_TARGET" ]]; then
    BUILD_DIR="$CARGO_OUTPUT_DIR/$RUST_TARGET/release"
else
    BUILD_DIR="$CARGO_OUTPUT_DIR/release"
fi

mkdir -p "$STAGING/bin" "$STAGING/frontend" "$STAGING/models" "$STAGING/data" "$STAGING/logs" "$STAGING/run" "$STAGING/backend/calibration_data"
for binary in billiards-server billiards-perception-live billiards-perception-gstreamer-probe; do
    require_file "$BUILD_DIR/$binary" "built Rust binary"
    install -m 0755 "$BUILD_DIR/$binary" "$STAGING/bin/$binary"
done

# ort's copy-dylibs feature may place ONNX Runtime beside the release binaries.
# Include the runtime library when present; GStreamer and OpenVINO remain native
# system dependencies so their platform plugins and device providers work.
shopt -s nullglob
for runtime_library in "$BUILD_DIR"/libonnxruntime*.dylib "$BUILD_DIR"/libonnxruntime*.so "$BUILD_DIR"/libonnxruntime*.so.*; do
    install -m 0644 "$runtime_library" "$STAGING/bin/$(basename -- "$runtime_library")"
done
shopt -u nullglob

install -m 0644 "$ONNX_SOURCE" "$STAGING/models/billiards-yolov8n.onnx"
install -m 0644 "$MANIFEST_SOURCE" "$STAGING/models/billiards-yolov8n.onnx.manifest.json"
install -m 0644 "$CHECKPOINT_SOURCE" "$STAGING/models/billiards-yolov8n.pt"

# Paths are deliberately relative to the package root so the entire bundle can
# be moved without a host-specific absolute model or executable path.
jq \
    --arg model 'models/billiards-yolov8n.onnx' \
    --arg checkpoint 'models/billiards-yolov8n.pt' \
    --arg live 'bin/billiards-perception-live' \
    --arg helper 'bin/billiards-perception-gstreamer-probe' \
    '.perception.rust_live = (.perception.rust_live // {}) |
     .perception.rust_live.onnx_path = $model |
     .perception.rust_live.source_checkpoint_path = $checkpoint |
     .perception.rust_live.executable_path = $live |
     .perception.rust_live.capture_helper_executable_path = $helper |
     .vision.camera.video_file_path = null' \
    "$CONFIG_SOURCE" > "$STAGING/config.json"
cp -R "$PROJECT_ROOT/backend/static/." "$STAGING/frontend/"
[[ -f "$STAGING/frontend/index.html" ]] || fail "packaged frontend is missing index.html after copying backend/static"
install -m 0755 "$SCRIPT_DIR/run.sh" "$STAGING/run.sh"
install -m 0755 "$SCRIPT_DIR/update.sh" "$STAGING/update.sh"

{
    printf 'Built UTC: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'Rust target: %s\n' "${RUST_TARGET:-host-native}"
    rustc -vV | sed -n 's/^host: /Rust host: /p'
    if git -C "$PROJECT_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
        printf 'Git revision: %s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
        if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
            printf '%s\n' 'Git worktree: dirty'
        else
            printf '%s\n' 'Git worktree: clean'
        fi
    fi
} > "$STAGING/build-info.txt"

# Replace the prior bundle only after every build and package step succeeds.
previous_dist=""
if [[ -e "$DIST_DIR" ]]; then
    previous_dist="${DIST_DIR}.previous.$(date -u '+%Y%m%dT%H%M%SZ')"
    mv -- "$DIST_DIR" "$previous_dist"
fi
if ! mv -- "$STAGING" "$DIST_DIR"; then
    if [[ -n "$previous_dist" && -e "$previous_dist" ]]; then
        mv -- "$previous_dist" "$DIST_DIR"
    fi
    fail "could not place the completed Rust distribution at $DIST_DIR"
fi
if [[ -n "$previous_dist" && -e "$previous_dist" ]]; then
    rm -rf -- "$previous_dist"
fi
trap - EXIT INT TERM

printf 'Rust distribution created: %s\n' "$DIST_DIR"
printf 'The bundle is native to %s; install matching GStreamer runtime/plugins on the destination machine.\n' "${RUST_TARGET:-the build host}"
printf 'Start it with: cd %q && ./run.sh\n' "$DIST_DIR"
