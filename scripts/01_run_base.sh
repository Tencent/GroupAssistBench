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

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <gid> <input.jsonl|input.xlsx> <run_dir> [workers] [rt2_count|all]"; exit 2
fi
GID="$1"; INPUT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; export PIPE_RUN_DIR="$3"
W="${4:-1}"; RT2="${5:-all}"
if [ ! -f "pipeline_runs/$PIPE_RUN_DIR/messages.jsonl" ]; then
  if [[ "$INPUT" == *.xlsx ]]; then
    if [ "${SKIP_PROFILE:-0}" = "1" ]; then
      "$PY" pipeline/ingest_xlsx.py "$INPUT" --gid "$GID"
    else
      "$PY" pipeline/ingest_xlsx.py "$INPUT" --gid "$GID" --profile
    fi
  else
    export PIPE_INPUT_JSONL="$INPUT"
    if [ "${SKIP_PROFILE:-0}" = "1" ]; then "$PY" pipeline/ingest.py "$GID" --no-profile; else "$PY" pipeline/ingest.py "$GID"; fi
  fi
fi
"$PY" pipeline/p2_locate.py "$GID" --workers "$W"
"$PY" pipeline/p3_anchor.py "$GID" --workers "$W"
"$PY" pipeline/p4_classify.py "$GID"
"$PY" pipeline/p4_layer.py "$GID" --workers "$W"
"$PY" pipeline/p4_d3.py "$GID"
"$PY" pipeline/p4_conflict.py "$GID" --full --workers "$W"
"$PY" pipeline/p5_build.py "$GID" --rt1 --rt2 "$RT2"
"$PY" pipeline/refresh_run_meta.py "$GID"
echo "base complete: pipeline_runs/$PIPE_RUN_DIR"
