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

if [ "$#" -lt 2 ]; then echo "usage: $0 <gid> <run_dir> [human_verdicts.jsonl]"; exit 2; fi
GID="$1"; RUN_DIR="$2"; HUMAN="${3:-}"; export PIPE_RUN_DIR="$RUN_DIR"
[ -f "pipeline_runs/$RUN_DIR/d3_pairs_r7.json" ] || { echo "missing d3_pairs_r7.json; run 02_screen_d3.sh first"; exit 2; }
# Re-running is safe: p5_build skips conflict pairs that already produced an item or a
# terminal "no material" outcome, so finished work is reused rather than duplicated.
# 可安全重跑: p5_build 会跳过已出题或已有取料结论的冲突对, 已完成的部分直接复用。
if [ -n "$HUMAN" ]; then
  "$PY" pipeline/r7_pairs_screen.py "$RUN_DIR" --human "$HUMAN" \
    --review-md "pipeline_runs/${RUN_DIR}/r7_pairs_review.md"
fi
"$PY" pipeline/p5_build.py "$GID" --rt2-d3 all
"$PY" pipeline/p5_build.py "$GID" --rt2-cf all
"$PY" pipeline/refresh_run_meta.py "$GID"
"$PY" tools/audit_canonical.py "$RUN_DIR"
echo "final pipeline complete: pipeline_runs/$RUN_DIR"
