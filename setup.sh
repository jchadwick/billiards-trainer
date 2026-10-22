#!/usr/bin/env bash
# Prepare the Rust server and browser client development dependencies.
# System packages are intentionally left to the user's package manager.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

missing=()
for command_name in cargo rustc node npm make; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        missing+=("$command_name")
    fi
done

if ((${#missing[@]} > 0)); then
    printf 'Missing required development tools: %s\n' "${missing[*]}" >&2
    printf 'Install the Rust toolchain, Node.js with npm, and GNU Make, then rerun ./setup.sh.\n' >&2
    exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0; then
    printf 'GStreamer development libraries were not detected. The default camera-enabled build needs them.\n' >&2
    printf 'Install GStreamer development packages for your operating system before running make build or make run.\n' >&2
fi

make -C "$repo_root" dev

cat <<'EOF'

Development setup is complete.

Next steps:
  1. Check or prepare config.json with the model and capture settings.
  2. Run `make test` to verify the Rust server and browser client.
  3. Run `make run` to start the local application.

The retained Python compatibility and model tools are optional. Install their
dependencies with `make install-python-tools` and run their checks with
`make test-python`.
EOF
