#!/usr/bin/env bash
# Train four LIBERO suites (20k steps each) — use when single-task pipeline is stable.
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in libero_spatial_suite libero_object_suite libero_goal_suite libero_long_suite; do
  echo "=== Training ${cfg} ==="
  python -m miniwam.train --config "configs/${cfg}.yaml" "$@"
done
