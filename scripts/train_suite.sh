#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m miniwam.train --config configs/libero_object_suite.yaml "$@"
