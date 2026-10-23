#!/usr/bin/env bash
# Verify that the Rust security audit fails closed without cargo-audit and
# dispatches the expected cargo subcommand when the audit plugin is available.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/billiards-security-check.XXXXXX")"
trap 'rm -rf -- "$TEST_ROOT"' EXIT INT TERM

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

MAKE_BIN="$(command -v make)"
EMPTY_PATH="$TEST_ROOT/empty-path"
FAKE_BIN="$TEST_ROOT/fake-bin"
BUILD_LOG="$TEST_ROOT/security-check.log"
AUDIT_MARKER="$TEST_ROOT/cargo-audit-invoked"
mkdir -p "$EMPTY_PATH" "$FAKE_BIN"

if PATH="$EMPTY_PATH" "$MAKE_BIN" -C "$PROJECT_ROOT" security-check-rust \
    >"$BUILD_LOG" 2>&1; then
    fail 'Rust security check passed without cargo-audit'
fi
grep -q 'cargo-audit is required' "$BUILD_LOG" || {
    cat "$BUILD_LOG" >&2
    fail 'missing-auditor failure did not explain the required tool'
}
if grep -q 'cargo audit' "$BUILD_LOG"; then
    fail 'Rust audit command ran after the auditor was found missing'
fi

cat > "$FAKE_BIN/cargo-audit" <<'SCRIPT'
#!/bin/sh
exit 0
SCRIPT
cat > "$FAKE_BIN/cargo" <<'SCRIPT'
#!/bin/sh
set -eu
if [ "$#" -ne 1 ] || [ "$1" != audit ]; then
    printf 'unexpected cargo arguments: %s\n' "$*" >&2
    exit 64
fi
: > "$CARGO_AUDIT_MARKER"
SCRIPT
chmod 0755 "$FAKE_BIN/cargo-audit" "$FAKE_BIN/cargo"

PATH="$FAKE_BIN" CARGO_AUDIT_MARKER="$AUDIT_MARKER" \
    "$MAKE_BIN" -C "$PROJECT_ROOT" security-check-rust >"$BUILD_LOG" 2>&1 || {
    cat "$BUILD_LOG" >&2
    fail 'Rust security check did not invoke cargo audit with the plugin available'
}
[[ -f "$AUDIT_MARKER" ]] || fail 'cargo audit stub was not invoked'

printf '%s\n' 'security-check Rust audit failure and dispatch smoke passed'
