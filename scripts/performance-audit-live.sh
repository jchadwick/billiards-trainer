#!/usr/bin/env bash

# Build and run the Rust state audit with a managed Rust application server.
# Startup, readiness, measurements, reports, and child cleanup live in Rust.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
manifest="${repo_root}/perception/Cargo.toml"

if [[ -n "${CARGO_BUILD_TARGET:-}" ]]; then
  echo "performance-audit-live.sh must use a native build target to run locally" >&2
  exit 2
fi

target_dir="${CARGO_TARGET_DIR:-${repo_root}/perception/target}"
cargo build --release --locked --manifest-path "${manifest}" \
  --features camera-gstreamer-source \
  --bin billiards-performance-audit-run \
  --bin billiards-server \
  --bin billiards-perception-live \
  --bin billiards-perception-gstreamer-probe

exec "${target_dir}/release/billiards-performance-audit-run" \
  --server-executable "${target_dir}/release/billiards-server" "$@"
