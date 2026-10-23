#!/usr/bin/env bash

# Thin development entry point: repeat/warm-up/report behavior lives in Rust.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

exec cargo run --locked \
  --manifest-path "${repo_root}/perception/Cargo.toml" \
  --bin billiards-performance-audit-run -- "$@"
