#!/usr/bin/env bash
# Keep the historical setup entry point working while sharing the root setup.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repo_root/setup.sh" "$@"
