#!/usr/bin/env python3
# Recompute the pipeline section of run_meta.json from the artifacts on disk, so that
# runs that call the stages one by one (bypassing run_pipeline.py) still end with
# accurate statistics. Optionally also back-fills provenance fields on existing items.
# Usage (package root):
#   python3 pipeline/refresh_run_meta.py <gid> [--retrofit] [--dry-run]
#     --retrofit  also rewrite `track` / `criteria_origin` of existing items from
#                 messages.jsonl and the contract (the original is backed up as .bak)
#     --dry-run   print only, write nothing
# 根据磁盘上的产物重新统计 run_meta.json 的 pipeline 段; 分阶段运行时也能得到准确统计。
import json, os, sys, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.llm import load_jsonl, code_commit
from pipeline.p2_locate import rundir


def stats(d):
    cands = load_jsonl(os.path.join(d, 'candidates.jsonl'))
    anchors = load_jsonl(os.path.join(d, 'anchors.jsonl'))
    cls = load_jsonl(os.path.join(d, 'classify.jsonl'))
    items = load_jsonl(os.path.join(d, 'items.jsonl'))
    gates = load_jsonl(os.path.join(d, 'gates.jsonl'))
    arb = load_jsonl(os.path.join(d, 'arbitration_queue.jsonl'))
    usage = load_jsonl(os.path.join(d, 'usage.jsonl'))
    inv = []
    p_inv = os.path.join(d, 'anchor_inventory.json')
    if os.path.exists(p_inv):
        inv = json.load(open(p_inv, encoding='utf-8'))
    return {
        'candidates': len(cands),
        'candidates_by_route': dict(Counter(c.get('route') for c in cands)),
        'grounding': dict(Counter(str((c.get('grounding') or {}).get('v')) for c in cands)),
        'anchored_positions': len(anchors),
        'anchors_total': sum(len(r.get('anchors') or []) for r in anchors),
        'anchors_rejected': sum(len(r.get('anchors_rejected') or []) for r in anchors),
        'inventory': len(inv),
        'inventory_consumed': sum(1 for a in inv if a.get('consumed_by_items')),
        'classified': len(cls),
        'items': len(items),
        'items_by_status': dict(Counter(it.get('status') for it in items)),
        'items_by_layer': dict(Counter(it.get('layer') for it in items)),
        'items_by_difficulty': dict(Counter(it.get('difficulty') for it in items)),
        'items_by_track': dict(Counter(it.get('track') for it in items)),
        'items_by_query_origin': dict(Counter(it.get('query_origin') for it in items)),
        'gate_findings': dict(Counter(f.get('severity') for f in gates)),
        'gate_errors': sum(1 for f in gates if f.get('severity') == 'error'),
        'arbitration_queue': len(arb),
        'arbitration_by_stage': dict(Counter(a.get('stage') for a in arb)),
        'llm_calls': len(usage),
        'llm_calls_by_stage': dict(Counter(x.get('stage') for x in usage)),
        'tokens_in': sum(x.get('in', 0) for x in usage),
        'tokens_out': sum(x.get('out', 0) for x in usage),
        'code_commit': code_commit(),
        'refreshed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'refreshed_by': 'refresh_run_meta.py',
    }


def retrofit(d, dry):
    """Back-fill: `track` comes from src.track in messages.jsonl; `criteria_origin`
    follows the contract.
    补正存量字段: track 取自 messages.jsonl, criteria_origin 按 contract 规则。
    """
    msgs = load_jsonl(os.path.join(d, 'messages.jsonl'))
    track = next((m['src']['track'] for m in msgs
                  if isinstance(m.get('src'), dict) and m['src'].get('track')), None)
    p = os.path.join(d, 'items.jsonl')
    items = load_jsonl(p)
    if not items:
        return {}
    ch = Counter()
    for it in items:
        if track and it.get('track') != track:
            it['track'] = track
            ch['track'] += 1
        want = 'autogen' if it.get('layer') == 'L2' else 'annotation'
        if it.get('criteria_origin') != want:
            it['criteria_origin'] = want
            ch['criteria_origin'] += 1
    if ch and not dry:
        os.replace(p, p + '.bak')
        with open(p, 'w', encoding='utf-8') as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + '\n')
    return dict(ch)


def main():
    args = sys.argv[1:]
    gid = next((a for a in args if not a.startswith('-')), 'demo')
    dry = '--dry-run' in args
    d = rundir(gid)
    p_meta = os.path.join(d, 'run_meta.json')
    meta = json.load(open(p_meta, encoding='utf-8')) if os.path.exists(p_meta) else {'gid': gid}

    if '--retrofit' in args:
        ch = retrofit(d, dry)
        print(f'存量补正: {ch or "无需改动"}')

    st = stats(d)
    old = meta.get('pipeline') or {}
    for k in ('candidates', 'anchored_positions', 'items', 'gate_errors'):
        if old.get(k) is not None and old.get(k) != st.get(k):
            print(f'  {k}: {old.get(k)} → {st.get(k)}')
    meta['pipeline'] = {**{k: v for k, v in old.items() if k in ('mode', 'run_at')}, **st}
    if not dry:
        json.dump(meta, open(p_meta, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(meta['pipeline'], ensure_ascii=False, indent=1))
    print(('(dry-run, 未落盘) ' if dry else '') + f'→ {p_meta}')


if __name__ == '__main__':
    main()
