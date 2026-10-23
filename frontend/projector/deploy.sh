#!/usr/bin/env bash

# Local convenience entry point retained for existing developer workflows.
# This script never connects to or changes another machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'USAGE'
Usage: ./deploy.sh [run] [LÖVE arguments...]

Local-only commands:
  run [love arguments...]       Run the native projector on this machine.

Install LÖVE with this machine's package manager as needed. This script does
not install software, request credentials, or deploy.
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
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
