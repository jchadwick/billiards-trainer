#!/usr/bin/env bash

# Thin development entry point for bounded native trajectory receipt summaries.
# The JSON Lines protocol is validated and reduced by Rust.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

exec cargo run --locked \
  --manifest-path "${repo_root}/perception/Cargo.toml" \
  --bin billiards-performance-audit-trajectory-summary -- "$@"
