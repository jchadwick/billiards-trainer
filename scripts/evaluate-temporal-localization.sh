#!/usr/bin/env bash

# Thin development entry point for temporal detector-accuracy scoring.
# Rust owns the annotation, scene-stream, matching, and report logic.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

exec cargo run --locked \
  --manifest-path "${repo_root}/perception/Cargo.toml" \
  --bin billiards-temporal-evaluate -- "$@"
