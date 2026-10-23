#!/usr/bin/env bash
# Check the saved-video MJPEG fixture restarts after a disconnect and reaps its
# listener process when the wrapper receives a shutdown signal.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/billiards-mjpeg-fixture.XXXXXX")"
WRAPPER_PID=""

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [[ -n "$WRAPPER_PID" ]] && kill -0 "$WRAPPER_PID" 2>/dev/null; then
        kill -TERM "$WRAPPER_PID" 2>/dev/null || true
        wait "$WRAPPER_PID" 2>/dev/null || true
    fi
    local child_pid_file="$TEST_ROOT/child.pid"
    if [[ -f "$child_pid_file" ]]; then
        local child_pid
        child_pid="$(cat "$child_pid_file")"
        kill -KILL "$child_pid" 2>/dev/null || true
    fi
    rm -rf -- "$TEST_ROOT"
}
trap cleanup EXIT INT TERM

FAKE_FFMPEG="$TEST_ROOT/fake-ffmpeg"
VIDEO_FILE="$TEST_ROOT/input.mp4"
INVOCATION_COUNT="$TEST_ROOT/invocations"
WRAPPER_LOG="$TEST_ROOT/wrapper.log"
touch "$VIDEO_FILE"

cat > "$FAKE_FFMPEG" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
count=0
if [[ -f "$INVOCATION_COUNT" ]]; then
    count="$(cat "$INVOCATION_COUNT")"
fi
count=$((count + 1))
printf '%s\n' "$count" > "$INVOCATION_COUNT"

# Model a client disconnect: the first listener exits, causing the fixture to
# reopen the one-client FFmpeg listener. The next listener stays alive until
# the wrapper asks it to stop.
if ((count == 1)); then
    exit 1
fi
printf '%s\n' "$$" > "$TEST_ROOT/child.pid"
trap 'touch "$TEST_ROOT/child-stopped"; exit 0' TERM
while :; do
    sleep 0.05
done
SCRIPT
chmod 0755 "$FAKE_FFMPEG"

INVOCATION_COUNT="$INVOCATION_COUNT" TEST_ROOT="$TEST_ROOT" \
    BILLIARDS_TEST_VIDEO="$VIDEO_FILE" \
    BILLIARDS_TEST_VIDEO_PORT=18001 \
    BILLIARDS_TEST_VIDEO_RESTART_DELAY_SECONDS=0.05 \
    FFMPEG_BIN="$FAKE_FFMPEG" \
    bash "$PROJECT_ROOT/scripts/serve-test-video-mjpeg.sh" >"$WRAPPER_LOG" 2>&1 &
WRAPPER_PID=$!

attempt=0
while [[ ! -f "$TEST_ROOT/child.pid" ]] && ((attempt < 200)); do
    if ! kill -0 "$WRAPPER_PID" 2>/dev/null; then
        cat "$WRAPPER_LOG" >&2
        fail 'MJPEG fixture exited before restarting its listener'
    fi
    sleep 0.025
    attempt=$((attempt + 1))
done
[[ -f "$TEST_ROOT/child.pid" ]] || {
    cat "$WRAPPER_LOG" >&2
    fail 'MJPEG fixture did not start the replacement listener'
}
[[ "$(cat "$INVOCATION_COUNT")" == 2 ]] || fail 'fixture did not restart exactly once after the simulated disconnect'

child_pid="$(cat "$TEST_ROOT/child.pid")"
kill -TERM "$WRAPPER_PID"
if wait "$WRAPPER_PID"; then
    wrapper_status=0
else
    wrapper_status=$?
fi
WRAPPER_PID=""
[[ "$wrapper_status" == 0 ]] || {
    cat "$WRAPPER_LOG" >&2
    fail "MJPEG fixture returned status $wrapper_status during graceful shutdown"
}
[[ -f "$TEST_ROOT/child-stopped" ]] || fail 'fixture did not wait for the active listener to handle shutdown'
if kill -0 "$child_pid" 2>/dev/null; then
    fail "listener process $child_pid is still alive after wrapper shutdown"
fi

printf '%s\n' 'saved-video MJPEG restart and child shutdown smoke passed'
