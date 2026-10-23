#!/usr/bin/env bash

# Build and run the Rust external-subscriber audit in release mode. Subscriber
# overhead is part of the measurement, so avoid debug-build parsing/JSON costs.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
manifest="${repo_root}/perception/Cargo.toml"

if [[ -n "${CARGO_BUILD_TARGET:-}" ]]; then
  echo "performance-audit-run.sh must use a native build target to run locally" >&2
  exit 2
fi

target_dir="${CARGO_TARGET_DIR:-${repo_root}/perception/target}"
cargo build --release --locked --manifest-path "${manifest}" \
  --bin billiards-performance-audit-run

exec "${target_dir}/release/billiards-performance-audit-run" "$@"
