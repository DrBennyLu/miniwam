#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Start LIBERO server in another terminal first:
#   cd /home/lxg/ai_models/LIBERO && python scripts/libero_ws_server.py --suite libero_object
python -m miniwam.eval.libero_ws_eval --config configs/libero_object_mini.yaml "$@"
