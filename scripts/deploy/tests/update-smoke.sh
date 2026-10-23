#!/usr/bin/env bash
# Exercise update.sh with disposable processes and a fake service manager.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/billiards-update.XXXXXX")"
trap 'cleanup' EXIT INT TERM

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    local pid
    if [[ -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ "$pid" =~ ^[0-9]+$ ]]; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    for pid in "${BACKGROUND_PIDS[@]:-}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    rm -rf -- "$TEST_ROOT"
}

process_is_running() {
    local pid="$1"
    local state
    kill -0 "$pid" 2>/dev/null || return 1
    state="$(ps -p "$pid" -o stat= 2>/dev/null || true)"
    [[ "$state" != Z* ]]
}

assert_run_count() {
    local expected="$1"
    local actual=0
    if [[ -f "$RUN_LOG" ]]; then
        actual="$(wc -l < "$RUN_LOG" | tr -d '[:space:]')"
    fi
    [[ "$actual" == "$expected" ]] || fail "expected run.sh call count $expected, got $actual"
}

start_server() {
    local mode="$1"
    (
        SERVER_MODE="$mode" nohup "$SERVER" >/dev/null 2>&1 &
        printf '%s\n' "$!" > "$PID_FILE"
    ) &
    local starter=$!
    BACKGROUND_PIDS+=("$starter")
    wait "$starter"
    local pid
    pid="$(cat "$PID_FILE")"
    for _ in {1..40}; do
        if process_is_running "$pid"; then
            return 0
        fi
        sleep 0.05
    done
    fail "fixture server did not start: $pid"
}

run_update() {
    local path="$1"
    shift
    env \
        PATH="$path:$PATH" \
        BILLIARDS_SERVER_EXECUTABLE="$SERVER" \
        BILLIARDS_PID_FILE="$PID_FILE" \
        UPDATE_RUN_LOG="$RUN_LOG" \
        SYSTEMD_LOG="$SYSTEMD_LOG" \
        "$@" \
        bash "$UPDATE_SCRIPT" > "$UPDATE_LOG" 2>&1
}

UPDATE_SCRIPT="$TEST_ROOT/scripts/deploy/update.sh"
SERVER="$TEST_ROOT/bin/billiards-server"
PID_FILE="$TEST_ROOT/run/billiards-server.pid"
RUN_LOG="$TEST_ROOT/run-calls.log"
SYSTEMD_LOG="$TEST_ROOT/systemd-calls.log"
UPDATE_LOG="$TEST_ROOT/update.log"
FAKE_BIN="$TEST_ROOT/fake-bin"
BACKGROUND_PIDS=()
mkdir -p "$(dirname "$UPDATE_SCRIPT")" "$(dirname "$SERVER")" "$(dirname "$PID_FILE")" "$FAKE_BIN"
ln -s /bin/bash "$FAKE_BIN/bash"
cp "$PROJECT_ROOT/scripts/deploy/update.sh" "$UPDATE_SCRIPT"

cat > "$TEST_ROOT/scripts/deploy/run.sh" <<'SCRIPT'
#!/usr/bin/env bash
printf 'run\n' >> "$UPDATE_RUN_LOG"
SCRIPT

cat > "$SERVER" <<'SCRIPT'
#!/usr/bin/env bash
if [[ "${SERVER_MODE:-graceful}" == ignore ]]; then
    trap '' TERM
else
    trap 'exit 0' TERM
fi
while :; do
    sleep 1 &
    wait "$!"
done
SCRIPT

cat > "$FAKE_BIN/systemctl" <<'SCRIPT'
#!/usr/bin/env bash
case "${1:-}" in
    cat) [[ "${SYSTEMD_PRESENT:-false}" == true ]] ;;
    restart)
        printf 'restart\n' >> "$SYSTEMD_LOG"
        [[ "${SYSTEMD_RESTART_FAIL:-false}" != true ]]
        ;;
    *) exit 64 ;;
esac
SCRIPT
chmod 0755 "$TEST_ROOT/scripts/deploy/run.sh" "$SERVER" "$FAKE_BIN/systemctl"
: > "$RUN_LOG"
: > "$SYSTEMD_LOG"

# A matching server gets a graceful TERM, its PID file is removed, and run.sh
# starts exactly once.
start_server graceful
SERVER_PID="$(cat "$PID_FILE")"
run_update "$FAKE_BIN" BILLIARDS_STOP_TIMEOUT_SECONDS=3 BILLIARDS_FORCE_STOP_TIMEOUT_SECONDS=1
! process_is_running "$SERVER_PID" || fail 'gracefully stopped server is still running'
[[ ! -e "$PID_FILE" ]] || fail 'PID file remains after graceful stop'
assert_run_count 1

# A matching server that ignores TERM is killed after the configured grace
# interval, and run.sh still starts only once after observed process exit.
start_server ignore
SERVER_PID="$(cat "$PID_FILE")"
run_update "$FAKE_BIN" BILLIARDS_STOP_TIMEOUT_SECONDS=1 BILLIARDS_FORCE_STOP_TIMEOUT_SECONDS=2
! process_is_running "$SERVER_PID" || fail 'SIGKILL escalation did not stop the server'
[[ ! -e "$PID_FILE" ]] || fail 'PID file remains after forced stop'
assert_run_count 2

# An unrelated PID is never signalled and prevents a replacement server.
(
    nohup sleep 60 >/dev/null 2>&1 &
    printf '%s\n' "$!" > "$TEST_ROOT/unrelated.pid"
) &
UNRELATED_STARTER=$!
BACKGROUND_PIDS+=("$UNRELATED_STARTER")
wait "$UNRELATED_STARTER"
UNRELATED_PID="$(cat "$TEST_ROOT/unrelated.pid")"
printf '%s\n' "$UNRELATED_PID" > "$PID_FILE"
if run_update "$FAKE_BIN"; then
    fail 'updater accepted a live unrelated PID'
fi
process_is_running "$UNRELATED_PID" || fail 'updater signalled an unrelated process'
[[ ! -s "$RUN_LOG" || "$(wc -l < "$RUN_LOG" | tr -d '[:space:]')" == 2 ]] || fail 'updater started run.sh after refusing an unrelated PID'
kill -TERM "$UNRELATED_PID" 2>/dev/null || true
rm -f "$PID_FILE"

# A completed process and malformed PID file are stale state and are removed.
true &
STALE_PID=$!
wait "$STALE_PID"
printf '%s\n' "$STALE_PID" > "$PID_FILE"
run_update "$FAKE_BIN"
[[ ! -e "$PID_FILE" ]] || fail 'stale PID file was not removed'
assert_run_count 3
printf 'not-a-pid\n' > "$PID_FILE"
run_update "$FAKE_BIN"
[[ ! -e "$PID_FILE" ]] || fail 'malformed PID file was not removed'
assert_run_count 4

# Invalid timeout values fail before touching the PID file or launching.
printf 'not-a-pid\n' > "$PID_FILE"
if run_update "$FAKE_BIN" BILLIARDS_FORCE_STOP_TIMEOUT_SECONDS=0; then
    fail 'updater accepted a zero forced-stop timeout'
fi
[[ -f "$PID_FILE" ]] || fail 'invalid timeout modified the PID file'
assert_run_count 4
rm -f "$PID_FILE"

# A detected service manager is authoritative. Failure must not fall through
# to the unmanaged launcher.
run_update "$FAKE_BIN" SYSTEMD_PRESENT=true
assert_run_count 4
[[ "$(wc -l < "$SYSTEMD_LOG" | tr -d '[:space:]')" == 1 ]] || fail 'systemd restart was not requested exactly once'
if run_update "$FAKE_BIN" SYSTEMD_PRESENT=true SYSTEMD_RESTART_FAIL=true; then
    fail 'updater hid a failed systemd restart'
fi
assert_run_count 4
[[ "$(wc -l < "$SYSTEMD_LOG" | tr -d '[:space:]')" == 2 ]] || fail 'failed systemd restart unexpectedly fell back or retried'

# Simulate an uninterruptible process by making the script's kill calls report
# success without delivering signals. The updater must not launch a second
# server when the PID remains live after both bounded waits.
start_server ignore
SERVER_PID="$(cat "$PID_FILE")"
cat > "$TEST_ROOT/no-kill.bash" <<'SCRIPT'
kill() {
    if [[ "${1:-}" == -0 ]]; then
        builtin kill "$@"
    else
        return 0
    fi
}
SCRIPT
if run_update "$FAKE_BIN" BASH_ENV="$TEST_ROOT/no-kill.bash" BILLIARDS_STOP_TIMEOUT_SECONDS=1 BILLIARDS_FORCE_STOP_TIMEOUT_SECONDS=1; then
    fail 'updater started a replacement while the old process remained alive'
fi
process_is_running "$SERVER_PID" || fail 'live-process fixture ended before the fail-closed check'
assert_run_count 4
kill -KILL "$SERVER_PID" 2>/dev/null || true
rm -f "$PID_FILE"

printf '%s\n' 'update graceful/forced shutdown, PID ownership, stale state, and systemd smoke passed'
