#!/usr/bin/env bash
# Build a native Rust server distribution with the retained static web client.
# The build host and packaged target must have compatible Rust and GStreamer
# toolchains. Inference uses a verified ONNX artifact.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
CONFIG_SOURCE="${BILLIARDS_CONFIG_SOURCE:-$PROJECT_ROOT/config.json}"
ONNX_SOURCE="${BILLIARDS_ONNX_MODEL:-}"
CHECKPOINT_SOURCE="${BILLIARDS_SOURCE_CHECKPOINT:-}"
DIST_DIR="${BILLIARDS_DIST_DIR:-$PROJECT_ROOT/dist}"
RUST_TARGET="${BILLIARDS_RUST_TARGET:-}"
RUST_FEATURES="${BILLIARDS_RUST_FEATURES:-camera-gstreamer-source}"
ONNX_MANIFEST_SOURCE="${ONNX_SOURCE}.manifest.json"
DISTRIBUTION_MARKER_SOURCE="$SCRIPT_DIR/distribution-marker.txt"

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
require_file "$DISTRIBUTION_MARKER_SOURCE" "distribution ownership marker"
require_file "$CONFIG_SOURCE" "Rust application configuration"
[[ -n "$CHECKPOINT_SOURCE" ]] || fail "set BILLIARDS_SOURCE_CHECKPOINT to the source .pt checkpoint matching the selected model manifest"
require_file "$CHECKPOINT_SOURCE" "source model checkpoint"

# The selected inference provider remains owned by the source configuration.
# Packaging only chooses which verified model artifacts to include and writes
# their package-relative paths into the staged configuration.
INFERENCE_PROVIDER="$(jq -er '.perception.rust_live.provider // "openvino"' "$CONFIG_SOURCE")" \
    || fail "configuration must define a valid perception.rust_live.provider: $CONFIG_SOURCE"
case "$INFERENCE_PROVIDER" in
    openvino)
        [[ -n "$ONNX_SOURCE" ]] || fail "set BILLIARDS_ONNX_MODEL to the verified ONNX model to include in the distribution"
        require_file "$ONNX_SOURCE" "ONNX model"
        require_file "$ONNX_MANIFEST_SOURCE" "ONNX export manifest"
        ;;
    *)
        fail "unsupported perception.rust_live.provider '$INFERENCE_PROVIDER' in $CONFIG_SOURCE"
        ;;
esac

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
' "$ONNX_MANIFEST_SOURCE" >/dev/null || fail "ONNX export manifest has an unsupported or incomplete schema: $ONNX_MANIFEST_SOURCE"
expected_onnx_sha="$(jq -er '.exported_model.sha256 | ascii_downcase' "$ONNX_MANIFEST_SOURCE")"
expected_checkpoint_sha="$(jq -er '.source_model.sha256 | ascii_downcase' "$ONNX_MANIFEST_SOURCE")"
actual_onnx_sha="$(sha256_file "$ONNX_SOURCE")"
actual_checkpoint_sha="$(sha256_file "$CHECKPOINT_SOURCE")"
[[ "$actual_onnx_sha" == "$expected_onnx_sha" ]] || fail "ONNX file digest does not match its export manifest: $ONNX_SOURCE"
[[ "$actual_checkpoint_sha" == "$expected_checkpoint_sha" ]] || fail "source checkpoint digest does not match its ONNX export manifest: $CHECKPOINT_SOURCE"

pkg-config --exists gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0 || fail "GStreamer development libraries are required for camera-gstreamer-source (gstreamer-1.0, gstreamer-app-1.0, and gstreamer-video-1.0)"

if [[ "$DIST_DIR" != /* ]]; then
    DIST_DIR="$PROJECT_ROOT/$DIST_DIR"
fi
DIST_NAME="$(basename -- "$DIST_DIR")"
[[ -n "$DIST_NAME" && "$DIST_NAME" != "." && "$DIST_NAME" != ".." && "$DIST_NAME" != "/" ]] || fail "refusing unsafe distribution output path: $DIST_DIR"
mkdir -p "$(dirname -- "$DIST_DIR")"
DIST_PARENT="$(cd -- "$(dirname -- "$DIST_DIR")" && pwd -P)"
DIST_DIR="$DIST_PARENT/$DIST_NAME"
[[ "$DIST_DIR" != "/" && "$DIST_DIR" != "$PROJECT_ROOT" ]] || fail "refusing unsafe distribution output path: $DIST_DIR"
[[ ! -L "$DIST_DIR" ]] || fail "refusing to replace a symbolic-link distribution path: $DIST_DIR"
if [[ -e "$DIST_DIR" ]]; then
    [[ -d "$DIST_DIR" ]] || fail "existing distribution output is not a directory: $DIST_DIR"
    [[ -f "$DIST_DIR/.billiards-distribution" && -r "$DIST_DIR/.billiards-distribution" ]] || fail "existing output is not a recognized Billiards distribution; move it aside before building: $DIST_DIR"
    [[ "$(cat -- "$DIST_DIR/.billiards-distribution")" == "$(cat -- "$DISTRIBUTION_MARKER_SOURCE")" ]] || fail "existing output has an unsupported Billiards distribution marker: $DIST_DIR"
fi
STAGING="$(mktemp -d "$DIST_PARENT/.${DIST_NAME}.build.XXXXXX")"
cleanup() {
    rm -rf -- "$STAGING"
}
trap cleanup EXIT INT TERM

printf '%s\n' 'Building retained HTML/JS/CSS client...'
(
    cd "$PROJECT_ROOT/frontend/web"
    npm ci
    # Build directly into the package staging directory. The normal Vite
    # output is backend/static, which the running development server may be
    # serving; an interrupted build must not partially replace those assets.
    npm run build -- --outDir "$STAGING/frontend"
)
[[ -f "$STAGING/frontend/index.html" ]] || fail "frontend build completed without a staged index.html"

printf '%s\n' 'Building Rust server and live capture binaries...'
(
    cd "$PROJECT_ROOT"
    if [[ -n "$RUST_TARGET" ]]; then
        cargo build \
            --locked \
            --manifest-path "$PROJECT_ROOT/perception/Cargo.toml" \
            --release \
            --features "$RUST_FEATURES" \
            --target "$RUST_TARGET" \
            --bin billiards-server \
            --bin billiards-perception-live \
            --bin billiards-perception-gstreamer-probe
    else
        cargo build \
            --locked \
            --manifest-path "$PROJECT_ROOT/perception/Cargo.toml" \
            --release \
            --features "$RUST_FEATURES" \
            --bin billiards-server \
            --bin billiards-perception-live \
            --bin billiards-perception-gstreamer-probe
    fi
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

mkdir -p "$STAGING/bin" "$STAGING/models" "$STAGING/data" "$STAGING/logs" "$STAGING/run" "$STAGING/backend/calibration_data"
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
install -m 0644 "$ONNX_MANIFEST_SOURCE" "$STAGING/models/billiards-yolov8n.onnx.manifest.json"
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
     .vision.camera.video_file_path = ""' \
    "$CONFIG_SOURCE" > "$STAGING/config.json"
# Localization profiles are bounded data assets, not model files. Copy exact
# bytes and rewrite only configuration paths so moving the bundle cannot leave
# it dependent on files in the build workspace. Runtime validates their schema,
# estimator revision and capture context before using them.
localization_limit="$(jq -er '.perception.rust_live.ball_localization.maximum_asset_bytes // 4194304 | select(type == "number" and floor == . and . > 0)' "$CONFIG_SOURCE")" \
    || fail "ball localization maximum_asset_bytes must be a positive integer"
for asset_key in optics_profile_path ball_center_calibration_path; do
    asset_source="$(jq -er --arg key "$asset_key" '.perception.rust_live.ball_localization[$key] // "" | select(type == "string")' "$CONFIG_SOURCE")" \
        || fail "ball localization $asset_key must be a string or null"
    [[ -n "$asset_source" ]] || continue
    [[ "$asset_source" == /* ]] || asset_source="$PROJECT_ROOT/$asset_source"
    require_file "$asset_source" "ball localization $asset_key"
    asset_size="$(wc -c < "$asset_source" | tr -d '[:space:]')"
    (( asset_size <= localization_limit )) || fail "ball localization $asset_key exceeds maximum_asset_bytes ($localization_limit)"
    jq -e 'type == "object"' "$asset_source" >/dev/null || fail "ball localization $asset_key must contain a JSON object"
    asset_destination="backend/calibration_data/ball-localization/$asset_key.json"
    mkdir -p "$STAGING/backend/calibration_data/ball-localization"
    install -m 0644 "$asset_source" "$STAGING/$asset_destination"
    jq --arg key "$asset_key" --arg path "$asset_destination" \
        '.perception.rust_live.ball_localization[$key] = $path' \
        "$STAGING/config.json" > "$STAGING/config.next.json"
    mv "$STAGING/config.next.json" "$STAGING/config.json"
done

printf '%s\n' "$INFERENCE_PROVIDER" > "$STAGING/.billiards-inference-provider"
[[ -f "$STAGING/frontend/index.html" ]] || fail "packaged frontend is missing index.html after the frontend build"
install -m 0755 "$SCRIPT_DIR/run.sh" "$STAGING/run.sh"
install -m 0755 "$SCRIPT_DIR/update.sh" "$STAGING/update.sh"
install -m 0644 "$DISTRIBUTION_MARKER_SOURCE" "$STAGING/.billiards-distribution"

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

# Replace only a recognized prior bundle after every build and package step
# succeeds. Keep its unique backup after success because it may contain the
# only copy of runtime configuration or calibration data.
previous_dist=""
if [[ -e "$DIST_DIR" ]]; then
    previous_dist="$(mktemp -d "$DIST_PARENT/.${DIST_NAME}.previous.XXXXXX")"
    if ! mv -- "$DIST_DIR" "$previous_dist/distribution"; then
        rmdir -- "$previous_dist" 2>/dev/null || true
        fail "could not preserve the existing distribution before replacement: $DIST_DIR"
    fi
fi
if ! mv -- "$STAGING" "$DIST_DIR"; then
    if [[ -n "$previous_dist" && -e "$previous_dist" ]]; then
        if ! mv -- "$previous_dist/distribution" "$DIST_DIR"; then
            fail "could not install the completed distribution or restore the previous bundle; recovery copy remains at $previous_dist/distribution"
        fi
        rmdir -- "$previous_dist"
    fi
    fail "could not place the completed Rust distribution at $DIST_DIR"
fi
trap - EXIT INT TERM

printf 'Rust distribution created: %s\n' "$DIST_DIR"
if [[ -n "$previous_dist" ]]; then
    printf 'Previous distribution retained for recovery: %s/distribution\n' "$previous_dist"
fi
printf 'The bundle is native to %s; install matching GStreamer runtime/plugins on the destination machine.\n' "${RUST_TARGET:-the build host}"
printf 'Start it with: cd %q && ./run.sh\n' "$DIST_DIR"
