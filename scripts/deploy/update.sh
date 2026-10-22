#!/usr/bin/env bash
# Restart the local Rust application after replacing its distribution files.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${BILLIARDS_SERVER_EXECUTABLE:-${BILLIARDS_BIN_DIR:-$SCRIPT_DIR/bin}/billiards-server}"
PID_FILE="${BILLIARDS_PID_FILE:-${BILLIARDS_RUN_DIR:-$SCRIPT_DIR/run}/billiards-server.pid}"
STOP_TIMEOUT_SECONDS="${BILLIARDS_STOP_TIMEOUT_SECONDS:-10}"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ "$STOP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "BILLIARDS_STOP_TIMEOUT_SECONDS must be a positive integer"

# Prefer the service manager when this distribution is installed as a system
# service. A failed restart is reported instead of falling through to a second
# un-managed server process.
if command -v systemctl >/dev/null 2>&1 && systemctl cat billiards-trainer.service >/dev/null 2>&1; then
    printf 'Restarting billiards-trainer.service...\n'
    systemctl restart billiards-trainer.service || fail "systemd could not restart billiards-trainer.service; check service permissions and logs"
    exit 0
fi

if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
        case "$command_line" in
            *"$SERVER"*)
                printf 'Stopping Rust server process %s...\n' "$pid"
                kill -TERM "$pid" || fail "could not signal Rust server process $pid"
                deadline=$((SECONDS + STOP_TIMEOUT_SECONDS))
                while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do
                    sleep 0.2
                done
                if kill -0 "$pid" 2>/dev/null; then
                    kill -KILL "$pid" || fail "Rust server process $pid did not stop after $STOP_TIMEOUT_SECONDS seconds"
                fi
                ;;
            *)
                fail "PID file $PID_FILE refers to process $pid, not this Rust server; refusing to stop it"
                ;;
        esac
    fi
    # run.sh removes its PID file on a normal shutdown; remove a stale file if
    # the recorded process was already gone.
    if [[ -f "$PID_FILE" ]] && { ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; }; then
        rm -f -- "$PID_FILE"
    fi
fi

exec "$SCRIPT_DIR/run.sh"
