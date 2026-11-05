#!/usr/bin/env python3
# Orchestrator: P1 ingest -> P2 locate -> P3 anchor -> P4 assign -> P5 build, plus the
# deterministic gates. Each stage resumes on its own; this entry point only sequences them.
# P1 接入 → P2 定位 → P3 锚定 → P4 判定 → P5 构造(+ 确定性闸门)。各阶段自带断点续跑, 此处只负责调度。
#
# Two modes:
#   gold mode (--gold, mostly no API): P2/P3 take the human annotations directly; P4,
#     P5 and the gates run automatically. Useful for annotated groups and regression tests.
#   proposal mode (default, needs a model): P2/P3 are run by the model and their output
#     is marked status=proposal until it passes the gates. Use --estimate first.
# 两种模式: --gold 直接采用人工标注作 P2/P3 结果; 默认由模型完成 P2/P3。
#
# Usage:
#   python3 pipeline/run_pipeline.py <gid> --gold [--rt2 N]          # gold mode (RT2 still calls the model)
#   python3 pipeline/run_pipeline.py <gid> [--limit N] [--workers 8]  # proposal mode
#   python3 pipeline/run_pipeline.py <gid> --estimate                 # cost estimate only
import json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.contract import ROOT
from pipeline.llm import config_banner, load_jsonl, code_commit

PY = sys.executable


def sh(args):
    print(f'\n$ {" ".join(args)}', flush=True)
    r = subprocess.run([PY] + args, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f'阶段失败: {" ".join(args)}')


def estimate(gid):
    from pipeline.p2_locate import rundir, load_msgs, enumerate_candidates
    d = rundir(gid)
    msgs = load_msgs(d)
    cands = enumerate_candidates(msgs)
    ctx_tokens = sum(len(m['text'] or '') for m in msgs) // 2  # rough estimate: about 2 Chinese characters per token
    n = len(cands)
    # P2: request check (short) + counterfactual probe (full context, then two short calls)
    p2_in = n * (200 + ctx_tokens + 300 + 600)
    # P3: anchor extraction (full context)
    p3_in = n * ctx_tokens
    # P5: query_specific for RT1-L2, assumed for half of the candidates
    p5_in = n // 2 * 800
    total_in = p2_in + p3_in + p5_in
    # Rough price: $3/M input + $10/M output, output assumed to be about 3% of input
    usd = total_in / 1e6 * 3 + total_in * 0.03 / 1e6 * 10
    print(json.dumps({'gid': gid, 'messages': len(msgs), 'candidates': n,
                      'ctx_tokens_per_call': ctx_tokens, 'est_input_tokens': total_in,
                      'est_usd(粗估,±2x)': round(usd, 2)}, ensure_ascii=False, indent=1))


def main():
    args = sys.argv[1:]
    gid = next((a for a in args if not a.startswith('-')), 'demo')
    gold = '--gold' in args
    n_rt2 = args[args.index('--rt2') + 1] if '--rt2' in args else 0  # passed through; accepts 'all'
    limit = ['--limit', args[args.index('--limit') + 1]] if '--limit' in args else []
    if '--workers' in args:  # per-candidate concurrency, passed to P2/P3
        limit += ['--workers', args[args.index('--workers') + 1]]
    if '--estimate' in args:
        return estimate(gid)

    print(config_banner(), flush=True)
    t0 = time.time()
    # P1 runs only if the run directory does not exist yet (ingest has no resume)
    from pipeline.p2_locate import rundir
    try:
        d = rundir(gid)
    except SystemExit:
        sh(['pipeline/ingest.py', gid])
        d = rundir(gid)

    if gold:
        sh(['pipeline/p2_locate.py', gid, '--enum-only', '--use-gold-grounding'] + limit)
        sh(['pipeline/p3_anchor.py', gid, '--from-gold'] + limit)
    else:
        sh(['pipeline/p2_locate.py', gid] + limit)
        sh(['pipeline/p3_anchor.py', gid] + limit)
    sh(['pipeline/p4_classify.py', gid])
    # Decide L1/L2 from the properties of the expected answer; only fills pending items.
    sh(['pipeline/p4_layer.py', gid] + limit)
    p5 = ['pipeline/p5_build.py', gid, '--rt1'] + limit
    if gold and not n_rt2:
        p5.append('--no-llm')
    if n_rt2:
        p5 += ['--rt2', str(n_rt2)]
    sh(p5)

    # Write the summary into run_meta
    meta_p = os.path.join(d, 'run_meta.json')
    meta = json.load(open(meta_p, encoding='utf-8')) if os.path.exists(meta_p) else {}
    items = load_jsonl(os.path.join(d, 'items.jsonl'))
    findings = load_jsonl(os.path.join(d, 'gates.jsonl'))
    arb = load_jsonl(os.path.join(d, 'arbitration_queue.jsonl'))
    usage = load_jsonl(os.path.join(d, 'usage.jsonl'))
    meta['pipeline'] = {
        'mode': 'gold(仲裁)' if gold else 'proposal(提案)',
        'code_commit': code_commit(),
        'candidates': len(load_jsonl(os.path.join(d, 'candidates.jsonl'))),
        'anchored_positions': len(load_jsonl(os.path.join(d, 'anchors.jsonl'))),
        'items': len(items),
        'items_by_status': {s: sum(1 for i in items if i['status'] == s)
                            for s in {i['status'] for i in items}},
        'gate_errors': sum(1 for f in findings if f['severity'] == 'error'),
        'arbitration_queue': len(arb),
        'llm_calls': len(usage),
        'wall_seconds': round(time.time() - t0, 1),
        'run_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json.dump(meta, open(meta_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n=== run_meta.pipeline ===')
    print(json.dumps(meta['pipeline'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
