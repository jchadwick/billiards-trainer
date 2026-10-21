#!/usr/bin/env bash

# Local convenience entry point retained for existing developer workflows.
# This script never connects to or changes another machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'USAGE'
Usage: ./deploy.sh [run|test|web] [arguments...]

Local-only commands:
  run [love arguments...]       Run the native projector on this machine.
  test [duration [interval]]     Run the local projector and UDP test sender.
  web                           Build the browser package locally.

Install LÖVE, Python, Node.js, and npm with this machine's package manager as
needed. This script does not install software, request credentials, or deploy.
USAGE
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$1" >&2
        exit 1
    fi
}

ACTION="${1:-run}"
if (($# > 0)); then
    shift
fi

case "$ACTION" in
    run)
        require_command love
        cd "$SCRIPT_DIR"
        exec love . "$@"
        ;;
    test)
        require_command love
        require_command python3
        exec "$SCRIPT_DIR/test_local.sh" "$@"
        ;;
    web)
        require_command node
        require_command npm
        exec "$SCRIPT_DIR/build_html.sh"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
