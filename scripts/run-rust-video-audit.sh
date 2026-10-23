#!/usr/bin/env bash

# Thin release-mode entry point. Stream inspection, bounded decoding, replay,
# inference, and report generation stay in the Rust executable.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

exec cargo run --release --locked \
  --manifest-path "${repo_root}/perception/Cargo.toml" \
  --bin billiards-perception-video-replay -- "$@"
