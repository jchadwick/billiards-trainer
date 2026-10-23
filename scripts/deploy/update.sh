#!/usr/bin/env bash
# Restart the local Rust application after replacing its distribution files.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${BILLIARDS_SERVER_EXECUTABLE:-${BILLIARDS_BIN_DIR:-$SCRIPT_DIR/bin}/billiards-server}"
PID_FILE="${BILLIARDS_PID_FILE:-${BILLIARDS_RUN_DIR:-$SCRIPT_DIR/run}/billiards-server.pid}"
STOP_TIMEOUT_SECONDS="${BILLIARDS_STOP_TIMEOUT_SECONDS:-10}"
FORCE_STOP_TIMEOUT_SECONDS="${BILLIARDS_FORCE_STOP_TIMEOUT_SECONDS:-2}"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ "$STOP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "BILLIARDS_STOP_TIMEOUT_SECONDS must be a positive integer"
[[ "$FORCE_STOP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "BILLIARDS_FORCE_STOP_TIMEOUT_SECONDS must be a positive integer"

process_is_running() {
    local pid="$1"
    local state
    kill -0 "$pid" 2>/dev/null || return 1
    state="$(ps -p "$pid" -o stat= 2>/dev/null || true)"
    # A zombie has exited and cannot serve requests, even if its parent has
    # not reaped it yet. Empty state means ps observed that the process had
    # disappeared between kill -0 and ps; unknown nonempty state fails closed.
    [[ -n "$state" && "$state" != Z* ]]
}

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
    if [[ "$pid" =~ ^[0-9]+$ ]] && process_is_running "$pid"; then
        command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
        case "$command_line" in
            *"$SERVER"*)
                printf 'Stopping Rust server process %s...\n' "$pid"
                if ! kill -TERM "$pid" 2>/dev/null && process_is_running "$pid"; then
                    fail "could not signal Rust server process $pid"
                fi
                deadline=$((SECONDS + STOP_TIMEOUT_SECONDS))
                while process_is_running "$pid" && (( SECONDS < deadline )); do
                    sleep 0.2
                done
                if process_is_running "$pid"; then
                    if ! kill -KILL "$pid" 2>/dev/null && process_is_running "$pid"; then
                        fail "Rust server process $pid did not stop after $STOP_TIMEOUT_SECONDS seconds"
                    fi
                    force_deadline=$((SECONDS + FORCE_STOP_TIMEOUT_SECONDS))
                    while process_is_running "$pid" && (( SECONDS < force_deadline )); do
                        sleep 0.2
                    done
                    process_is_running "$pid" && fail "Rust server process $pid remained alive after SIGKILL; refusing to start a replacement"
                fi
                ;;
            *)
                fail "PID file $PID_FILE refers to process $pid, not this Rust server; refusing to stop it"
                ;;
        esac
    fi
    # run.sh removes its PID file on a normal shutdown; remove a stale file if
    # the recorded process was already gone.
    if [[ -f "$PID_FILE" ]] && { ! [[ "$pid" =~ ^[0-9]+$ ]] || ! process_is_running "$pid"; }; then
        rm -f -- "$PID_FILE"
    fi
fi

exec "$SCRIPT_DIR/run.sh"
