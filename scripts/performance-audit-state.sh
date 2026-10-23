#!/usr/bin/env bash

# Thin development entry point: keep argument parsing and measurement logic in
# the Rust audit binary, while allowing Cargo to build it from any directory.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

exec cargo run --locked \
  --manifest-path "${repo_root}/perception/Cargo.toml" \
  --bin billiards-performance-audit -- "$@"
