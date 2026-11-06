#!/usr/bin/env bash
# Six everyday prompts through taskbot design, one line per result; transcripts are saved.
# Usage: scripts/smoke.sh [backend] [model] [extra args, e.g. --no-sim]
#   scripts/smoke.sh ollama qwen2.5:7b-instruct
set -euo pipefail
backend="${1:-ollama}"
model="${2:-qwen2.5:7b-instruct}"
shift $(( $# > 2 ? 2 : $# ))
exec python -m taskbot.smoke --backend "$backend" --model "$model" "$@"
