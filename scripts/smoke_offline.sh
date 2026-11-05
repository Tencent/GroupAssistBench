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

export PIPE_INPUT_JSONL="$ROOT/scripts/smoke_sample.jsonl"
export PIPE_RUN_DIR="demo_smoke"
rm -rf "$ROOT/pipeline_runs/$PIPE_RUN_DIR"
"$PY" pipeline/ingest.py demo --no-profile
"$PY" pipeline/p2_locate.py demo --enum-only
"$PY" - <<'CHECK'
import json, pathlib
p=pathlib.Path('pipeline_runs/demo_smoke')
msgs=[json.loads(x) for x in (p/'messages.jsonl').read_text(encoding='utf-8').splitlines() if x]
cands=[json.loads(x) for x in (p/'candidates.jsonl').read_text(encoding='utf-8').splitlines() if x]
assert len(msgs)==6, len(msgs)
assert cands and all('turn' in x and 'query' in x for x in cands)
print('offline smoke passed:', {'messages':len(msgs),'candidates':len(cands)})
CHECK
