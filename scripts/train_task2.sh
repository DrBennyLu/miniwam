#!/usr/bin/env bash
# task 2 单 task 联合 WAM 训练
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG="${1:-configs/libero_object_mini.yaml}"
RESUME="${2:-}"

if [[ -n "$RESUME" ]]; then
  python -m miniwam.train --config "$CONFIG" --resume "$RESUME"
else
  python -m miniwam.train --config "$CONFIG"
fi
