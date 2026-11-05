#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Interpreter: most distros ship only `python3`; override with PYTHON=/path/to/python.
# 解释器: 多数系统只有 python3; 需要指定虚拟环境时用 PYTHON=... 覆盖。
PY="${PYTHON:-python3}"

# Step 4: every finished run directory -> one evaluation bundle (all built items).
# Only items that are not built are removed; there is no sampling here (that is step 05).
# Steps 01-03 run per group; this step collects across groups into one bundle.
# Zero API, read-only on the runs, safe to re-run.
# 第 4 步: 把所有跑完的 run 目录合成一个评测包, 只剔除非 built 的题, 不抽样(抽样是第 5 步)。
# 01–03 逐群运行, 本步跨群汇总; 不调 API, 可反复重跑。
#
# usage: $0 [out_dir] [runs_dir]
#   out_dir   default eval_bundles/bundle_<YYYYMMDD>
#   runs_dir  default pipeline_runs/ ; use an absolute path to point elsewhere
#
# For the single-group demo the bundle goes to pipeline_runs/demo_4368/eval_bundle/;
# with several groups use the default eval_bundles/.
# 单群 demo 的评测包放在 pipeline_runs/demo_4368/eval_bundle/; 多群时用默认的 eval_bundles/。

OUT="${1:-eval_bundles/bundle_$(date +%Y%m%d)}"
RUNS="${2:-}"

ARGS=(--out "$OUT")
if [ -n "$RUNS" ]; then ARGS+=(--runs-dir "$RUNS"); fi

"$PY" tools/build_eval_bundle.py "${ARGS[@]}"

echo
echo "bundle complete: $OUT"
echo "  题目  : $OUT/items_adapted.jsonl"
echo "  群聊  : $OUT/contexts.json"
echo "  截断点: $OUT/cutoffs.json"
echo "  清单  : $OUT/manifest.json  (全部 built 题, 按群/层/难度分桶)"
echo
echo "只收 status=built。带 error 级闸门结论的题已被剔除, 见 bundle_meta.json 的 skipped。"
echo
echo "下一步(抽均衡子集):"
echo "  bash scripts/05_sample_manifest.sh $OUT 1500"
