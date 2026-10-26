#!/usr/bin/env bash
# Exercise the real package builder with isolated files and fake build tools.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/billiards-build-dist.XXXXXX")"
trap 'rm -rf -- "$TEST_ROOT"' EXIT INT TERM

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

assert_file() {
    [[ -f "$1" ]] || fail "expected file is missing: $1"
}

assert_no_staging_dirs() {
    local -a matches=()
    shopt -s nullglob
    matches=("$DIST_PARENT"/.distribution.build.*)
    shopt -u nullglob
    ((${#matches[@]} == 0)) || fail "builder left staging directories behind: ${matches[*]}"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

REPO="$TEST_ROOT/repository"
FAKE_BIN="$TEST_ROOT/fake-bin"
MV_BIN="$TEST_ROOT/failing-mv-bin"
TARGET_DIR="$TEST_ROOT/cargo-target"
DIST_PARENT="$TEST_ROOT/output"
DIST_DIR="$DIST_PARENT/distribution"
CONFIG_SOURCE="$TEST_ROOT/live-config.json"
ONNX_SOURCE="$TEST_ROOT/model.onnx"
CHECKPOINT_SOURCE="$TEST_ROOT/model.pt"
MANIFEST_SOURCE="$ONNX_SOURCE.manifest.json"
CORAL_MODEL_NAME="billiards-yolov8n_edgetpu.tflite"
CORAL_SOURCE="$TEST_ROOT/$CORAL_MODEL_NAME"
CORAL_MANIFEST_SOURCE="$TEST_ROOT/manifest.json"
CORAL_CONFIG_SOURCE="$TEST_ROOT/coral-config.json"
BUILD_LOG="$TEST_ROOT/build.log"
FAIL_MOVE_ONCE="$TEST_ROOT/fail-move-once"

mkdir -p "$REPO/scripts/deploy" "$REPO/frontend/web" "$REPO/backend/static" "$REPO/perception" "$FAKE_BIN" "$MV_BIN" "$DIST_PARENT"
ln -s /bin/bash "$FAKE_BIN/bash"
cp "$PROJECT_ROOT/scripts/deploy/build-dist.sh" "$REPO/scripts/deploy/build-dist.sh"
cp "$PROJECT_ROOT/scripts/deploy/clean-dist.sh" "$REPO/scripts/deploy/clean-dist.sh"
cp "$PROJECT_ROOT/scripts/deploy/run.sh" "$REPO/scripts/deploy/run.sh"
cp "$PROJECT_ROOT/scripts/deploy/update.sh" "$REPO/scripts/deploy/update.sh"
cp "$PROJECT_ROOT/scripts/deploy/distribution-marker.txt" "$REPO/scripts/deploy/distribution-marker.txt"
cp "$PROJECT_ROOT/Makefile" "$REPO/Makefile"
touch "$REPO/perception/Cargo.toml"
printf '<!doctype html><title>currently served client</title>\n' > "$REPO/backend/static/index.html"
printf 'preserve this build output\n' > "$REPO/backend/static/current-client.css"
SERVED_CLIENT_SHA="$(sha256_file "$REPO/backend/static/index.html")"
SERVED_ASSET_SHA="$(sha256_file "$REPO/backend/static/current-client.css")"

cat > "$CONFIG_SOURCE" <<'JSON'
{"vision":{"camera":{"video_file_path":"","device_id":"http://127.0.0.1:8001/stream.mjpg"}},"perception":{"capture":{}}}
JSON
printf 'fake onnx bytes\n' > "$ONNX_SOURCE"
printf 'fake checkpoint bytes\n' > "$CHECKPOINT_SOURCE"
ONNX_SHA="$(sha256_file "$ONNX_SOURCE")"
CHECKPOINT_SHA="$(sha256_file "$CHECKPOINT_SOURCE")"
cat > "$MANIFEST_SOURCE" <<JSON
{"manifest_schema_version":1,"exported_model":{"sha256":"$ONNX_SHA"},"source_model":{"sha256":"$CHECKPOINT_SHA"}}
JSON
printf 'fake compiled coral model bytes\n' > "$CORAL_SOURCE"
CORAL_SHA="$(sha256_file "$CORAL_SOURCE")"
cat > "$CORAL_MANIFEST_SOURCE" <<JSON
{
  "artifact_type": "Coral Edge TPU compiled TensorFlow Lite model",
  "source": {"checkpoint": {"sha256": "$CHECKPOINT_SHA"}},
  "model_contract": {
    "input": {"shape": [1, 1280, 1280, 3], "layout": "NHWC", "dtype": "float32", "color_order": "RGB", "normalization": "[0,1]"},
    "output": {"shape": [1, 6, 33600], "dtype": "float32", "class_order": ["cue", "ball"], "box_coordinate_space": "normalized_0_1", "nms_embedded": false}
  },
  "compiler_result": {"status": "success", "compiled_model": "$CORAL_MODEL_NAME"},
  "files": {"$CORAL_MODEL_NAME": {"sha256": "$CORAL_SHA"}}
}
JSON
jq '.perception.rust_live.provider = "coral"' "$CONFIG_SOURCE" > "$CORAL_CONFIG_SOURCE"

cat > "$FAKE_BIN/npm" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
case "$*" in
    ci) exit 0 ;;
    "run build -- --outDir "*)
        output_dir=""
        for argument in "$@"; do
            output_dir="$argument"
        done
        mkdir -p "$output_dir"
        if [[ "${FAKE_NPM_FAIL:-false}" == true ]]; then
            printf 'incomplete build output\n' > "$output_dir/index.html"
            printf 'simulated partial frontend build failure\n' >&2
            exit 72
        fi
        printf '<!doctype html><title>smoke client</title>\n' > "$output_dir/index.html"
        printf 'body { color: #123; }\n' > "$output_dir/client.css"
        ;;
    *) printf 'unexpected npm arguments: %s\n' "$*" >&2; exit 64 ;;
esac
SCRIPT

cat > "$FAKE_BIN/cargo" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${FAKE_CARGO_FAIL:-false}" == true ]]; then
    printf 'simulated Cargo failure\n' >&2
    exit 71
fi
output_root="$CARGO_TARGET_DIR"
while (($#)); do
    if [[ "$1" == --target ]]; then
        output_root="$output_root/$2"
        shift 2
    else
        shift
    fi
done
mkdir -p "$output_root/release"
for binary in billiards-server billiards-perception-live billiards-perception-gstreamer-probe; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "$output_root/release/$binary"
    chmod 0755 "$output_root/release/$binary"
done
printf 'fake onnx runtime\n' > "$output_root/release/libonnxruntime.so.1"
SCRIPT

cat > "$FAKE_BIN/rustc" <<'SCRIPT'
#!/usr/bin/env bash
printf 'rustc 1.99.0\nhost: smoke-test-target\n'
SCRIPT

cat > "$FAKE_BIN/node" <<'SCRIPT'
#!/usr/bin/env bash
exit 0
SCRIPT

cat > "$FAKE_BIN/pkg-config" <<'SCRIPT'
#!/usr/bin/env bash
exit 0
SCRIPT

cat > "$MV_BIN/mv" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -- && "${2:-}" == *".distribution.build."* && "${3:-}" == */distribution && ! -e "$FAIL_MOVE_ONCE" ]]; then
    : > "$FAIL_MOVE_ONCE"
    exit 71
fi
exec /bin/mv "$@"
SCRIPT
chmod 0755 "$FAKE_BIN/cargo" "$FAKE_BIN/node" "$FAKE_BIN/npm" "$FAKE_BIN/pkg-config" "$FAKE_BIN/rustc" "$MV_BIN/mv"

run_build() {
    local extra_path="${1:-$FAKE_BIN}"
    shift || true
    env \
        PATH="$extra_path:$PATH" \
        FAKE_REPOSITORY_ROOT="$REPO" \
        CARGO_TARGET_DIR="$TARGET_DIR" \
        BILLIARDS_CONFIG_SOURCE="$CONFIG_SOURCE" \
        BILLIARDS_ONNX_MODEL="$ONNX_SOURCE" \
        BILLIARDS_CORAL_MODEL= \
        BILLIARDS_SOURCE_CHECKPOINT="$CHECKPOINT_SOURCE" \
        BILLIARDS_DIST_DIR="$DIST_DIR" \
        "$@" \
        bash "$REPO/scripts/deploy/build-dist.sh" > "$BUILD_LOG" 2>&1
}

run_build "$FAKE_BIN"
assert_file "$DIST_DIR/.billiards-distribution"
assert_file "$DIST_DIR/bin/billiards-server"
assert_file "$DIST_DIR/bin/billiards-perception-live"
assert_file "$DIST_DIR/bin/billiards-perception-gstreamer-probe"
assert_file "$DIST_DIR/bin/libonnxruntime.so.1"
assert_file "$DIST_DIR/frontend/index.html"
assert_file "$DIST_DIR/frontend/client.css"
assert_file "$DIST_DIR/models/billiards-yolov8n.onnx"
assert_file "$DIST_DIR/models/billiards-yolov8n.onnx.manifest.json"
assert_file "$DIST_DIR/models/billiards-yolov8n.pt"
assert_file "$DIST_DIR/build-info.txt"
[[ "$(sha256_file "$REPO/backend/static/index.html")" == "$SERVED_CLIENT_SHA" ]] || fail 'package build overwrote the currently served index.html'
[[ "$(sha256_file "$REPO/backend/static/current-client.css")" == "$SERVED_ASSET_SHA" ]] || fail 'package build overwrote a currently served asset'
[[ -x "$DIST_DIR/bin/billiards-server" ]] || fail 'packaged Rust binaries must be executable'
[[ "$(jq -r '.perception.rust_live.onnx_path' "$DIST_DIR/config.json")" == models/billiards-yolov8n.onnx ]] || fail 'ONNX config path must be package-relative'
[[ "$(jq -r '.perception.rust_live.source_checkpoint_path' "$DIST_DIR/config.json")" == models/billiards-yolov8n.pt ]] || fail 'checkpoint config path must be package-relative'
[[ "$(jq -r '.perception.rust_live.executable_path' "$DIST_DIR/config.json")" == bin/billiards-perception-live ]] || fail 'live executable path must be package-relative'
[[ "$(jq -r '.perception.rust_live.capture_helper_executable_path' "$DIST_DIR/config.json")" == bin/billiards-perception-gstreamer-probe ]] || fail 'capture helper path must be package-relative'
[[ "$(jq -r '.vision.camera.video_file_path' "$DIST_DIR/config.json")" == '' ]] || fail 'packaged config must clear recorded-video input'
assert_no_staging_dirs

# An arbitrary directory must remain untouched and must not reach the build.
UNOWNED_DIR="$TEST_ROOT/unowned"
mkdir -p "$UNOWNED_DIR"
printf 'keep me\n' > "$UNOWNED_DIR/important.txt"
if env PATH="$FAKE_BIN:$PATH" FAKE_REPOSITORY_ROOT="$REPO" CARGO_TARGET_DIR="$TARGET_DIR" \
    BILLIARDS_CONFIG_SOURCE="$CONFIG_SOURCE" BILLIARDS_ONNX_MODEL="$ONNX_SOURCE" \
    BILLIARDS_SOURCE_CHECKPOINT="$CHECKPOINT_SOURCE" BILLIARDS_DIST_DIR="$UNOWNED_DIR" \
    bash "$REPO/scripts/deploy/build-dist.sh" > "$BUILD_LOG" 2>&1; then
    fail 'builder accepted an unmarked existing directory'
fi
[[ "$(cat "$UNOWNED_DIR/important.txt")" == 'keep me' ]] || fail 'builder changed unowned directory content'

UNOWNED_SYMLINK="$TEST_ROOT/unowned-link"
ln -s "$UNOWNED_DIR" "$UNOWNED_SYMLINK"
if env PATH="$FAKE_BIN:$PATH" FAKE_REPOSITORY_ROOT="$REPO" CARGO_TARGET_DIR="$TARGET_DIR" \
    BILLIARDS_CONFIG_SOURCE="$CONFIG_SOURCE" BILLIARDS_ONNX_MODEL="$ONNX_SOURCE" \
    BILLIARDS_SOURCE_CHECKPOINT="$CHECKPOINT_SOURCE" BILLIARDS_DIST_DIR="$UNOWNED_SYMLINK" \
    bash "$REPO/scripts/deploy/build-dist.sh" > "$BUILD_LOG" 2>&1; then
    fail 'builder accepted a symbolic-link output path'
fi
[[ -f "$UNOWNED_DIR/important.txt" ]] || fail 'builder modified data behind an output symlink'

WRONG_MARKER_DIR="$TEST_ROOT/wrong-marker"
mkdir -p "$WRONG_MARKER_DIR"
printf 'not a Billiards distribution\n' > "$WRONG_MARKER_DIR/.billiards-distribution"
if env PATH="$FAKE_BIN:$PATH" FAKE_REPOSITORY_ROOT="$REPO" CARGO_TARGET_DIR="$TARGET_DIR" \
    BILLIARDS_CONFIG_SOURCE="$CONFIG_SOURCE" BILLIARDS_ONNX_MODEL="$ONNX_SOURCE" \
    BILLIARDS_SOURCE_CHECKPOINT="$CHECKPOINT_SOURCE" BILLIARDS_DIST_DIR="$WRONG_MARKER_DIR" \
    bash "$REPO/scripts/deploy/build-dist.sh" > "$BUILD_LOG" 2>&1; then
    fail 'builder accepted a directory with an unsupported ownership marker'
fi

# A failed compiler invocation leaves the current package and its runtime data.
printf '{"important":true}\n' > "$DIST_DIR/data/runtime-state.json"
if run_build "$FAKE_BIN" FAKE_NPM_FAIL=true; then
    fail 'builder unexpectedly succeeded with a partially failing frontend build'
fi
grep -q 'simulated partial frontend build failure' "$BUILD_LOG" || {
    cat "$BUILD_LOG" >&2
    fail 'builder did not reach the simulated frontend failure'
}
[[ -f "$DIST_DIR/data/runtime-state.json" ]] || fail 'failed frontend build removed current runtime state'
[[ "$(sha256_file "$REPO/backend/static/index.html")" == "$SERVED_CLIENT_SHA" ]] || fail 'failed frontend build damaged the currently served index.html'
[[ "$(sha256_file "$REPO/backend/static/current-client.css")" == "$SERVED_ASSET_SHA" ]] || fail 'failed frontend build damaged a currently served asset'
assert_no_staging_dirs

if run_build "$FAKE_BIN" FAKE_CARGO_FAIL=true; then
    fail 'builder unexpectedly succeeded with a failing Cargo stub'
fi
grep -q 'simulated Cargo failure' "$BUILD_LOG" || fail 'builder did not reach the simulated Cargo failure'
[[ -f "$DIST_DIR/data/runtime-state.json" ]] || fail 'failed build removed current runtime state'
assert_no_staging_dirs

# A failed final rename restores the prior package and retains its data.
if run_build "$MV_BIN:$FAKE_BIN" FAIL_MOVE_ONCE="$FAIL_MOVE_ONCE"; then
    fail 'builder unexpectedly succeeded when final placement was rejected'
fi
[[ -f "$FAIL_MOVE_ONCE" ]] || fail 'the final placement failure was not injected'
grep -q 'could not place the completed Rust distribution' "$BUILD_LOG" || fail 'builder failed for a reason other than the injected final placement error'
[[ -f "$DIST_DIR/data/runtime-state.json" ]] || fail 'failed replacement did not restore runtime state'
[[ "$(cat "$DIST_DIR/.billiards-distribution")" == "$(cat "$REPO/scripts/deploy/distribution-marker.txt")" ]] || fail 'failed replacement did not restore the recognized package'
assert_no_staging_dirs

# Successful replacement preserves the old marked package under a unique path.
run_build "$FAKE_BIN"
BACKUP_DIRS=("$DIST_PARENT"/.distribution.previous.*)
[[ -d "${BACKUP_DIRS[0]:-}" ]] || fail 'successful replacement did not retain a previous bundle'
[[ -f "${BACKUP_DIRS[0]}/distribution/data/runtime-state.json" ]] || fail 'retained backup lost runtime data'
assert_no_staging_dirs

# Cleanup must reject unowned data, then remove only the marked distribution.
if BILLIARDS_DIST_DIR="$UNOWNED_DIR" bash "$REPO/scripts/deploy/clean-dist.sh" > "$BUILD_LOG" 2>&1; then
    fail 'cleanup accepted an unmarked directory'
fi
[[ -f "$UNOWNED_DIR/important.txt" ]] || fail 'cleanup removed unowned directory content'
if BILLIARDS_DIST_DIR="$UNOWNED_SYMLINK" bash "$REPO/scripts/deploy/clean-dist.sh" > "$BUILD_LOG" 2>&1; then
    fail 'cleanup accepted a symbolic-link distribution path'
fi
[[ -f "$UNOWNED_DIR/important.txt" ]] || fail 'cleanup removed data behind an output symlink'

# The broad development cleanup target must share the same ownership guard.
mkdir -p "$REPO/dist"
printf 'keep via make clean\n' > "$REPO/dist/important.txt"
if (cd "$REPO" && make clean) > "$BUILD_LOG" 2>&1; then
    fail 'make clean accepted an unmarked distribution directory'
fi
[[ -f "$REPO/dist/important.txt" ]] || fail 'make clean removed an unmarked distribution directory'

BILLIARDS_DIST_DIR="$DIST_DIR" bash "$REPO/scripts/deploy/clean-dist.sh" > "$BUILD_LOG" 2>&1
[[ ! -e "$DIST_DIR" ]] || fail 'cleanup did not remove the marked distribution'
[[ -f "${BACKUP_DIRS[0]}/distribution/data/runtime-state.json" ]] || fail 'cleanup removed the retained previous bundle'

# A Coral-selected config packages only the compiled model, its adjacent
# manifest, and the verified checkpoint. The runner preflight must accept this
# bundle without requiring the ONNX model trio.
run_build "$FAKE_BIN" \
    BILLIARDS_CONFIG_SOURCE="$CORAL_CONFIG_SOURCE" \
    BILLIARDS_ONNX_MODEL= \
    BILLIARDS_CORAL_MODEL="$CORAL_SOURCE"
assert_file "$DIST_DIR/models/$CORAL_MODEL_NAME"
assert_file "$DIST_DIR/models/manifest.json"
assert_file "$DIST_DIR/models/billiards-yolov8n.pt"
[[ ! -e "$DIST_DIR/models/billiards-yolov8n.onnx" ]] || fail 'Coral package unexpectedly includes an ONNX model'
[[ "$(jq -r '.perception.rust_live.provider' "$DIST_DIR/config.json")" == coral ]] || fail 'Coral package changed the configured inference provider'
[[ "$(jq -r '.perception.rust_live.coral_model_path' "$DIST_DIR/config.json")" == "models/$CORAL_MODEL_NAME" ]] || fail 'Coral model path must be package-relative'
[[ "$(jq -r '.perception.rust_live.onnx_path' "$DIST_DIR/config.json")" == '' ]] || fail 'Coral package retained an ONNX model path'
[[ "$(jq -r '.perception.rust_live.source_checkpoint_path' "$DIST_DIR/config.json")" == models/billiards-yolov8n.pt ]] || fail 'Coral checkpoint path must be package-relative'
[[ "$(cat "$DIST_DIR/.billiards-inference-provider")" == coral ]] || fail 'Coral provider metadata is missing'
bash "$DIST_DIR/run.sh" > "$BUILD_LOG" 2>&1 || {
    cat "$BUILD_LOG" >&2
    fail 'run.sh preflight rejected the Coral package'
}
[[ ! -e "$DIST_DIR/run/billiards-server.pid" ]] || fail 'successful Coral run.sh smoke left a stale PID file'

CORAL_PACKAGE_SHA="$(sha256_file "$DIST_DIR/models/$CORAL_MODEL_NAME")"
cp "$CORAL_SOURCE" "$TEST_ROOT/coral-model.original"
printf 'modified compiled model bytes\n' > "$CORAL_SOURCE"
if run_build "$FAKE_BIN" \
    BILLIARDS_CONFIG_SOURCE="$CORAL_CONFIG_SOURCE" \
    BILLIARDS_ONNX_MODEL= \
    BILLIARDS_CORAL_MODEL="$CORAL_SOURCE"; then
    fail 'builder accepted a compiled Coral model with a mismatched manifest digest'
fi
grep -q 'compiled Coral model digest does not match' "$BUILD_LOG" || fail 'builder did not reject the Coral model digest mismatch'
[[ "$(sha256_file "$DIST_DIR/models/$CORAL_MODEL_NAME")" == "$CORAL_PACKAGE_SHA" ]] || fail 'failed Coral model verification replaced the existing package'
assert_no_staging_dirs
cp "$TEST_ROOT/coral-model.original" "$CORAL_SOURCE"

cp "$CORAL_MANIFEST_SOURCE" "$TEST_ROOT/coral-manifest.original"
printf '{"artifact_type":"invalid"}\n' > "$CORAL_MANIFEST_SOURCE"
if run_build "$FAKE_BIN" \
    BILLIARDS_CONFIG_SOURCE="$CORAL_CONFIG_SOURCE" \
    BILLIARDS_ONNX_MODEL= \
    BILLIARDS_CORAL_MODEL="$CORAL_SOURCE"; then
    fail 'builder accepted an invalid Coral manifest'
fi
grep -q 'Coral export manifest is invalid' "$BUILD_LOG" || fail 'builder did not reject the invalid Coral manifest'
[[ "$(sha256_file "$DIST_DIR/models/$CORAL_MODEL_NAME")" == "$CORAL_PACKAGE_SHA" ]] || fail 'invalid Coral manifest replaced the existing package'
assert_no_staging_dirs
cp "$TEST_ROOT/coral-manifest.original" "$CORAL_MANIFEST_SOURCE"

cp "$CHECKPOINT_SOURCE" "$TEST_ROOT/checkpoint.original"
printf 'mismatched checkpoint bytes\n' > "$CHECKPOINT_SOURCE"
if run_build "$FAKE_BIN" \
    BILLIARDS_CONFIG_SOURCE="$CORAL_CONFIG_SOURCE" \
    BILLIARDS_ONNX_MODEL= \
    BILLIARDS_CORAL_MODEL="$CORAL_SOURCE"; then
    fail 'builder accepted a source checkpoint with a mismatched Coral manifest digest'
fi
grep -q 'source checkpoint digest does not match the Coral export manifest' "$BUILD_LOG" || fail 'builder did not reject the Coral checkpoint digest mismatch'
[[ "$(sha256_file "$DIST_DIR/models/$CORAL_MODEL_NAME")" == "$CORAL_PACKAGE_SHA" ]] || fail 'failed Coral checkpoint verification replaced the existing package'
assert_no_staging_dirs
cp "$TEST_ROOT/checkpoint.original" "$CHECKPOINT_SOURCE"

printf '%s\n' 'build-dist packaging, ownership, failure recovery, and cleanup smoke passed'
