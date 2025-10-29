#!/bin/bash
# Run geometric collision tests with proper PYTHONPATH
#
# Usage: ./run_geometric_collision_tests.sh

cd "$(dirname "$0")/../../.."
export PYTHONPATH="${PWD}/backend:${PWD}:${PYTHONPATH}"

python -m pytest backend/tests/core/test_geometric_collision.py -v "$@"
