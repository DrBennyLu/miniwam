#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
PY="${PY:-python}"
$PY scripts/smoke_vae.py --config configs/libero_object_mini_smoke.yaml
$PY -m miniwam.train --config configs/libero_object_mini_smoke.yaml
$PY -m miniwam.train --config configs/libero_object_no_video_smoke.yaml
$PY -m miniwam.train --config configs/libero_object_video_only_smoke.yaml
echo "All smoke tests passed."
