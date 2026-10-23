#!/usr/bin/env bash
# Remove only a distribution previously created by build-dist.sh.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
DIST_DIR="${BILLIARDS_DIST_DIR:-$PROJECT_ROOT/dist}"
MARKER_SOURCE="$SCRIPT_DIR/distribution-marker.txt"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ -f "$MARKER_SOURCE" && -r "$MARKER_SOURCE" && -s "$MARKER_SOURCE" ]] || fail "distribution ownership marker is missing: $MARKER_SOURCE"
if [[ "$DIST_DIR" != /* ]]; then
    DIST_DIR="$PROJECT_ROOT/$DIST_DIR"
fi

DIST_NAME="$(basename -- "$DIST_DIR")"
[[ -n "$DIST_NAME" && "$DIST_NAME" != "." && "$DIST_NAME" != ".." && "$DIST_NAME" != "/" ]] || fail "refusing unsafe distribution output path: $DIST_DIR"
DIST_PARENT_INPUT="$(dirname -- "$DIST_DIR")"
if [[ ! -d "$DIST_PARENT_INPUT" ]]; then
    printf 'No distribution exists at %s\n' "$DIST_DIR"
    exit 0
fi
DIST_PARENT="$(cd -- "$DIST_PARENT_INPUT" && pwd -P)"
DIST_DIR="$DIST_PARENT/$DIST_NAME"
[[ "$DIST_DIR" != "/" && "$DIST_DIR" != "$PROJECT_ROOT" ]] || fail "refusing unsafe distribution output path: $DIST_DIR"
[[ ! -L "$DIST_DIR" ]] || fail "refusing to remove a symbolic-link distribution path: $DIST_DIR"
[[ -e "$DIST_DIR" ]] || {
    printf 'No distribution exists at %s\n' "$DIST_DIR"
    exit 0
}
[[ -d "$DIST_DIR" ]] || fail "distribution path is not a directory: $DIST_DIR"
[[ -f "$DIST_DIR/.billiards-distribution" && -r "$DIST_DIR/.billiards-distribution" ]] || fail "refusing to remove an unrecognized directory: $DIST_DIR"
[[ "$(cat -- "$DIST_DIR/.billiards-distribution")" == "$(cat -- "$MARKER_SOURCE")" ]] || fail "refusing to remove a distribution with an unsupported marker: $DIST_DIR"

rm -rf -- "$DIST_DIR"
printf 'Removed recognized Billiards distribution: %s\n' "$DIST_DIR"
