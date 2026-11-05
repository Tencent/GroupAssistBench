#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Interpreter: most distros ship only `python3`; override with PYTHON=/path/to/python.
# 解释器: 多数系统只有 python3; 需要指定虚拟环境时用 PYTHON=... 覆盖。
PY="${PYTHON:-python3}"

# Step 5: draw a balanced evaluation subset from the full pool.
# Rules: one item per event cluster per cell; D3 kept in full; the other cells by quota,
# split across groups by sqrt(n). Uses construction-side properties only, never
# evaluation scores. Zero API, deterministic, no random seed.
# 第 5 步: 从全量题池抽出均衡子集。同事件簇同格只留 1 题, D3 全保, 其余格按配额抽;
# 只看构造侧属性, 不读评测分数, 结果完全确定。
#
# usage: $0 <bundle_dir> [target | per-cell] [runs_dir]
#   bundle_dir  output of step 04; the manifest is written to <bundle_dir>/manifest_sampled.json(l)
#   2nd arg     a number = the same target for every non-D3 cell (default 1500);
#               "L1D1=776,L1D2=648,L2D1=0,L2D2=650" = per-cell targets (0 keeps the whole cell)
#   runs_dir    default pipeline_runs/ ; must be the same as for step 04
#
# Examples:
#   bash scripts/05_sample_manifest.sh eval_bundles/bundle_20261102 1500
#   bash scripts/05_sample_manifest.sh pipeline_runs/demo_4368/eval_bundle 5     # demo

if [ "$#" -lt 1 ]; then echo "usage: $0 <bundle_dir> [target|per-cell] [runs_dir]"; exit 2; fi
BUNDLE="$1"; QUOTA="${2:-1500}"; RUNS="${3:-}"

if [ ! -f "$BUNDLE/items_adapted.jsonl" ]; then
  echo "找不到 $BUNDLE/items_adapted.jsonl —— 先跑 scripts/04_build_bundle.sh"; exit 2
fi

ARGS=(--out "$BUNDLE/manifest_sampled.jsonl")
if [[ "$QUOTA" == *=* ]]; then ARGS+=(--per-cell "$QUOTA"); else ARGS+=(--target "$QUOTA"); fi
if [ -n "$RUNS" ]; then ARGS+=(--src "$RUNS"); fi

"$PY" tools/sample_manifest.py "${ARGS[@]}"

echo
echo "sample complete:"
echo "  $BUNDLE/manifest_sampled.json   分桶清单, 评测 harness 用 --manifest 指向它"
echo "  $BUNDLE/manifest_sampled.jsonl  逐题明细 + 首行 _meta(配额与统计)"
