#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Interpreter: most distros ship only `python3`; override with PYTHON=/path/to/python.
# 解释器: 多数系统只有 python3; 需要指定虚拟环境时用 PYTHON=... 覆盖。
PY="${PYTHON:-python3}"
# Config file: defaults to .env; set ENV_FILE=/path/to/other.env to run a variant config.
ENV_FILE="${ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; fi

if [ "$#" -lt 1 ]; then echo "usage: $0 <run_dir> [workers]"; exit 2; fi
RUN_DIR="$1"; W="${2:-3}"
"$PY" pipeline/r7_pairs_screen.py "$RUN_DIR" --rounds 3 --workers "$W" \
  --review-md "pipeline_runs/${RUN_DIR}/r7_pairs_review.md"
echo "screen complete: pipeline_runs/$RUN_DIR/d3_pairs_r7.json"
echo "review sheet: pipeline_runs/${RUN_DIR}/r7_pairs_review.md"
