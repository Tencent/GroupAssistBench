#!/usr/bin/env python3
"""Stratified sampler: full instance pool -> a balanced evaluation manifest.
Zero API, fully deterministic, no random seed: the same inputs always give the same
manifest, so a published manifest can be regenerated and checked.

Why: constructed instances are very unevenly spread over the six level x dependency
cells, and one event cluster can yield many instances. Running the full pool is costly
and lets large groups and near-repeats dominate stratified results.

Seven-step funnel:
  1 version gate   only run directories produced by the current pipeline
  2 quality gate   status=built
  3 family dedup   one instance per event cluster (event_id, else the anchor set) per
                   L x D cell, native queries preferred
  4 scarce cells   both D3 cells (L1D3, L2D3) are kept in full
  5 cell quotas    TARGET for each of the other four cells, split across groups by
                   sqrt(n) (large groups are damped, small ones are not zeroed)
  6 in-cell fill   round-robin over group x cell x memory-span tercile; order within a
                   stratum = pairing first (the l1_pair of selected L2 items) ->
                   quality (fewer gate warnings, fewer provisional criteria, query
                   length not extreme) -> iid
  7 protocol       the manifest stores ids only; unselected instances stay in the pool

Only construction-side properties are used (cell, event cluster, gate warnings, memory
span, pairing); no evaluation score is read, so anyone can reproduce it on their own data.

Usage:
  python3 tools/sample_manifest.py --target 1500
  python3 tools/sample_manifest.py --per-cell "L1D1=776,L1D2=648,L2D1=0,L2D2=650"
  python3 tools/sample_manifest.py --target 1500 --min-items 30
  python3 tools/sample_manifest.py --src /path/to/runs --out manifest.jsonl

Two outputs with the same stem:
  <out>.jsonl   one row per selected instance, plus a leading _meta row (settings,
                quotas, statistics)
  <out>.json    bucketed ids {group/level/difficulty: [iid, ...]}, same shape as
                build_eval_bundle's manifest.json, usable with a harness's --manifest

分层抽样器: 从全量题池抽出均衡的评测清单。完全确定、不调 API、不读任何评测分数。
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent  # .../tools
ROOT = HERE.parent  # package root
SRC = ROOT / 'pipeline_runs'
RULES = 'stratified_v2(√n配额/配对优先/质量排序/D3全保)'

# L2 cells first (including D3), so want_l1 is complete before the L1 quotas are filled
CELLS = [('L2', 'D3'), ('L2', 'D1'), ('L2', 'D2'),
         ('L1', 'D3'), ('L1', 'D1'), ('L1', 'D2')]


def _rel(p):
    """Express a path relative to the package root when possible; outside it, keep only the
    directory name.
    """
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return f'<external>/{p.name}'


def span_of(it):
    t = [a['turn'] for a in it.get('anchors') or []]
    return (it.get('cut_turn') or 0) - min(t) if t else 0


def gname(dirname, items=None):
    """Group id: the gid field of items.jsonl, with the directory name only as a fallback
    (same rule as tools/build_eval_bundle.py; a directory such as demo_4368 does not start
    with the gid).
    群标识优先取 items.jsonl 的 gid 字段, 目录名只作兜底。
    """
    import re
    if items:
        gid = next((str(r['gid']) for r in items if r.get('gid')), None)
        if gid:
            return gid
    return re.sub(r'_20\d{6}[a-z]*(_[a-z0-9.]+)?$', '', dirname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=1500,
                    help='非 D3 各格的目标题数（D3 恒全保留）')
    ap.add_argument('--per-cell', default=None,
                    help='逐格目标，如 "L1D1=1500,L1D2=1500,L2D2=1500"；'
                         '设为 0 表示该格全保留')
    ap.add_argument('--min-items', type=int, default=0,
                    help='群题数下限，低于此值的群整群排除')
    ap.add_argument('--pair-ratio', default=None,
                    help='指定格的「有搭档题」最低占比，如 "L2D2=0.8"。'
                         '仅对 L2 格有效（配对方向 L2→L1）。')
    ap.add_argument('--d3-cap-update-nopair', default=None,
                    help='D3 中「update 且无搭档」的保留上限，按质量+难度取前 N。'
                         '格式 "L1=50,L2=60"（也接受单个数字，两格同值）。'
                         'arbitration(真冲突) 恒全保。')
    ap.add_argument('--out', default=None)
    ap.add_argument('--src', default=str(SRC))
    args = ap.parse_args()

    # --d3-cap-update-nopair: "L1=50,L2=60" or "50"
    d3cap = {}
    if args.d3_cap_update_nopair:
        v = str(args.d3_cap_update_nopair)
        if '=' in v:
            for kv in v.split(','):
                k2, n2 = kv.split('=')
                d3cap[k2.strip().upper()] = int(n2)
        else:
            d3cap = {'L1': int(v), 'L2': int(v)}

    pair_ratio = {}
    if args.pair_ratio:
        for kv in args.pair_ratio.split(','):
            k, v = kv.split('=')
            pair_ratio[k.strip()] = float(v)

    per_cell = {}
    if args.per_cell:
        for kv in args.per_cell.split(','):
            k, v = kv.split('=')
            per_cell[k.strip()] = int(v)

    src = Path(args.src)
    out = Path(args.out) if args.out else (
        ROOT / 'eval_bundles' / f'manifest_sampled_{time.strftime("%Y%m%d")}.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)

    pool = defaultdict(lambda: defaultdict(list))
    warns = Counter()
    n_src = n_built = 0
    skipped_small = []

    for d in sorted(x for x in src.iterdir() if x.is_dir()):
        f = d / 'items.jsonl'
        if not f.exists():
            continue
        lines = [l for l in open(f, encoding='utf-8') if l.strip()]
        # the gid is known only after reading items (see gname())
        g = gname(d.name, [json.loads(l) for l in lines])
        if args.min_items and len(lines) < args.min_items:
            skipped_small.append((g, len(lines)))
            continue
        # gates.jsonl may be missing
        gf = d / 'gates.jsonl'
        if gf.exists():
            for l in open(gf, encoding='utf-8'):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if r.get('severity') == 'warn':
                    warns[r['iid']] += 1
        fams = defaultdict(list)
        for l in lines:
            try:
                it = json.loads(l)
            except Exception:
                continue
            n_src += 1
            if it.get('status') != 'built':
                continue
            n_built += 1
            it['_g'] = g
            fam = (('ev', it['event_id']) if it.get('event_id') is not None
                   else ('as', tuple(sorted(a['turn']
                                            for a in it.get('anchors') or []))))
            fams[((it['layer'], it['difficulty']), fam)].append(it)
        for (cell, fam), its in fams.items():
            its.sort(key=lambda x: (x['query_origin'] != 'native', x['iid']))
            rep = its[0]
            rep['_fam'] = f'{fam[0]}:{fam[1]}'
            pool[cell][g].append(rep)

    def quality_key(it, want):
        crits = it.get('criteria') or []
        prov = (sum(1 for c in crits if c.get('provisional')) / len(crits)
                if crits else 0.0)
        long_q = 1 if len(it.get('query') or '') > 400 else 0
        return (it['iid'] not in want, warns.get(it['iid'], 0),
                round(prov, 2), long_q, it['iid'])

    # iid index of each L1 difficulty, used to check whether an L2 item has a partner
    l1_ids = {D: {i['iid'] for v in pool[('L1', D)].values() for i in v}
              for D in ('D1', 'D2', 'D3')}
    l2_ids = {D: {i['iid'] for v in pool[('L2', D)].values() for i in v}
              for D in ('D1', 'D2', 'D3')}

    picked, want_l1, rows = set(), set(), []
    for cell in CELLS:
        groups = pool[cell]
        if not groups:
            continue
        key = f'{cell[0]}{cell[1]}'
        tgt = per_cell.get(key, None)
        # D3 is always kept in full; an explicit per_cell value of 0 also means keep all
        if cell[1] == 'D3' or tgt == 0:
            quota = {g: len(v) for g, v in groups.items()}
        else:
            t = tgt if tgt is not None else args.target
            w = {g: math.sqrt(len(v)) for g, v in groups.items()}
            W = sum(w.values()) or 1
            quota = {g: min(len(groups[g]), round(t * w[g] / W))
                     for g in groups}
        # ── L2D3: arbitration-type items are kept in full; unpaired update-type items are cut by quality ──
        # Arbitration (a real multi-party conflict) is scarce and valuable, so it is always kept.
        # Update-type items without an L1 partner add nothing to pairing analyses and are
        # simpler, so only the best N are kept.
        # L2D3: 仲裁型全部保留; 无搭档的更新型按质量取前 N 道。
        if cell[1] == 'D3' and cell[0] in d3cap:
            # pairing direction: L2 looks up l1_pair in the L1 pool; L1 looks up l2_pair in the L2 pool
            pk = 'l1_pair' if cell[0] == 'L2' else 'l2_pair'
            tgt_ids = (l1_ids if cell[0] == 'L2' else l2_ids)['D3']
            keep, drop = [], []
            for g2, its2 in groups.items():
                for it in its2:
                    ck = it.get('conflict_kind')
                    hp = bool([p for p in (it.get(pk) or []) if p in tgt_ids])
                    (keep if (ck == 'arbitration' or hp) else drop).append(it)
            # unpaired update items: quality first, then difficulty (more anchors / longer span = harder)
            drop.sort(key=lambda x: (warns.get(x['iid'], 0),
                                     -len(x.get('anchors') or []),
                                     -span_of(x),
                                     quality_key(x, want_l1)))
            allow = {x['iid'] for x in drop[:d3cap[cell[0]]]}
            keep_ids = {k['iid'] for k in keep} | allow
            groups = {g2: [it for it in its2 if it['iid'] in keep_ids]
                      for g2, its2 in groups.items()}
            groups = {g2: v for g2, v in groups.items() if v}
            quota = {g2: len(v) for g2, v in groups.items()}

        # ── Pairing ratio: at least `ratio` of the cell's selection must have a partner ──
        # Ordering alone is not enough, because group quotas and span terciles often put the
        # partner in another stratum, so the quota is enforced explicitly here.
        # 配对配比: 显式保证该格中有搭档的题不低于 ratio。
        ratio = pair_ratio.get(key)
        if ratio and cell[0] == 'L2':
            tgt_pair = {g2: int(round(quota.get(g2, 0) * ratio))
                        for g2 in groups}
        else:
            tgt_pair = {}

        # ── L1 back-fill: partners of selected L2 items are taken first ──
        # Pair coverage counts a pair only when both members are selected, so each L1 cell first
        # takes want_l1 within the group's pool, then fills the rest as usual.
        # L1 侧先收入已选 L2 题的搭档, 再按常规流程填满。
        force = {}
        if cell[0] == 'L1' and want_l1:
            for g2, its2 in groups.items():
                f = [it for it in its2 if it['iid'] in want_l1]
                if f:
                    f.sort(key=lambda x: quality_key(x, want_l1))
                    # never more than the group's quota
                    force[g2] = f[:quota.get(g2, 0)]

        for g in sorted(groups):
            its = sorted(groups[g], key=span_of)
            if force.get(g):
                sel = list(force[g])
                rest = [x for x in its if x not in sel]
                terc = [sorted(b, key=lambda x: quality_key(x, want_l1))
                        for b in (rest[0::3], rest[1::3], rest[2::3])]
                i, guard = 0, 0
                while len(sel) < quota[g] and guard < 10000:
                    b = terc[i % 3]
                    if b:
                        sel.append(b.pop(0))
                    i += 1
                    guard += 1
                    if not any(terc):
                        break
            elif tgt_pair.get(g):
                # fill the quota with paired items by quality first, then the rest
                yes = [x for x in its
                       if [p for p in (x.get('l1_pair') or [])
                           if p in l1_ids.get(cell[1], set())]]
                no = [x for x in its if x not in yes]
                yes.sort(key=lambda x: quality_key(x, want_l1))
                sel = yes[:tgt_pair[g]]
                rest = [x for x in its if x not in sel]
                # the remainder still rotates over span terciles to keep memory spans balanced
                terc = [sorted(b, key=lambda x: quality_key(x, want_l1))
                        for b in (rest[0::3], rest[1::3], rest[2::3])]
                i, guard = 0, 0
                while len(sel) < quota[g] and guard < 10000:
                    b = terc[i % 3]
                    if b:
                        sel.append(b.pop(0))
                    i += 1
                    guard += 1
                    if not any(terc):
                        break
            else:
                terc = [sorted(b, key=lambda x: quality_key(x, want_l1))
                        for b in (its[0::3], its[1::3], its[2::3])]
                sel, i, guard = [], 0, 0
                while len(sel) < quota[g] and guard < 10000:
                    b = terc[i % 3]
                    if b:
                        sel.append(b.pop(0))
                    i += 1
                    guard += 1
                    if not any(terc):
                        break
            for it in sel:
                picked.add(it['iid'])
                if it['layer'] == 'L2':
                    want_l1 |= set(it.get('l1_pair') or [])
                rows.append({'iid': it['iid'], 'gid': it.get('gid'),
                             'group': g, 'layer': it['layer'],
                             'difficulty': it['difficulty'],
                             'query_origin': it.get('query_origin'),
                             'span': span_of(it), 'fam': it['_fam'],
                             'warns': warns.get(it['iid'], 0)})

    by_iid = {r['iid']: r for r in rows}
    for cell in CELLS:
        for g, its in pool[cell].items():
            for it in its:
                if it['iid'] in by_iid:
                    pairs = ((it.get('l1_pair') or []) +
                             (it.get('l2_pair') or []))
                    by_iid[it['iid']]['pair_in_set'] = sorted(
                        set(pairs) & picked)

    meta = {'_meta': {'rules': RULES, 'target': args.target,
                      'per_cell': per_cell or None,
                      'pair_ratio': pair_ratio or None,
                      'd3_cap_update_nopair': d3cap or None,
                      'min_items': args.min_items,
                      # Paths are stored relative to the package root, not as absolute paths: they would
                      # expose the local directory layout and make _meta from two machines incomparable.
                      'src': _rel(src), 'src_items': n_src, 'built': n_built,
                      'n_groups': len(set(r['group'] for r in rows)),
                      'selected': len(rows),
                      'skipped_small': skipped_small,
                      'generated_at': time.strftime('%F %T')}}
    with open(out, 'w', encoding='utf-8') as f:
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        for r in sorted(rows, key=lambda x: x['iid']):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Also write the bucketed manifest in the shape a harness's --manifest expects, using
    # the same rules as build_eval_bundle's manifest.json, so the two never drift apart.
    # 同时写出分桶清单, 规则与 build_eval_bundle 的 manifest.json 一致。
    bucket = defaultdict(list)
    for r in sorted(rows, key=lambda x: x['iid']):
        bucket[f"{r['group']}/{r['layer']}/{r['difficulty']}"].append(r['iid'])
    out_json = out.with_suffix('.json')
    json.dump(dict(sorted(bucket.items())), open(out_json, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    c = Counter((r['layer'], r['difficulty']) for r in rows)
    byg = Counter(r['group'] for r in rows)
    l2 = [r for r in rows if r['layer'] == 'L2']
    pair = sum(1 for r in l2 if r.get('pair_in_set'))
    fam_n = sum(len(v) for cl in pool.values() for v in cl.values())
    print(f'采样 {len(rows):,} 题 · 源 {n_src:,} · built {n_built:,} · '
          f'家族去重后 {fam_n:,}')
    print('格分布:', ' · '.join(f'{l}{d}={c[(l, d)]}'
                                for l in ('L1', 'L2')
                                for d in ('D1', 'D2', 'D3')))
    print(f'群数 {len(byg)} · 最大群占比 '
          f'{max(byg.values()) / max(len(rows), 1):.1%}')
    if l2:
        print(f'L2 配对保留: {pair}/{len(l2)} ({pair / len(l2):.0%})')
    if skipped_small:
        print(f'排除小群 {len(skipped_small)}: '
              f'{[g for g, _ in skipped_small]}')
    print(f'→ {out}')
    print(f'→ {out.with_suffix(".json")}  (分桶清单, harness --manifest 直接可用)')


if __name__ == '__main__':
    main()
