#!/usr/bin/env bash
# Validate the supported Rust and browser development toolchain.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

failures=0

check_command() {
    local command_name="$1"
    if command -v "$command_name" >/dev/null 2>&1; then
        printf '✓ %-8s %s\n' "$command_name" "$("$command_name" --version 2>&1 | head -n 1)"
    else
        printf '✗ %s is not installed\n' "$command_name" >&2
        failures=$((failures + 1))
    fi
}

printf 'Checking Rust and browser development tools\n'
check_command cargo
check_command rustc
check_command node
check_command npm

if ((failures == 0)); then
    cargo metadata --locked --manifest-path perception/Cargo.toml --no-deps >/dev/null
    if [[ -d frontend/web/node_modules ]]; then
        printf '✓ Browser dependencies are installed\n'
    else
        printf '✗ Browser dependencies are missing; run `make dev`\n' >&2
        failures=$((failures + 1))
    fi
fi

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0; then
    printf '✓ GStreamer development libraries are available\n'
else
    printf '⚠ GStreamer development libraries were not detected; the default camera-enabled build will need them\n' >&2
fi

if ((failures > 0)); then
    printf '\nDevelopment setup has %d required-tool issue(s).\n' "$failures" >&2
    exit 1
fi

printf '\nRust and browser development prerequisites are ready.\n'
