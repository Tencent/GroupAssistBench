#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run directories -> evaluation bundle.

This is the single interface between the construction pipeline and any evaluation
harness. Pure code, zero API calls, standard library only. It:
  1. discovers groups automatically: a group whose run has finished P5 is included;
  2. maps contract fields to harness fields (iid / kind / group / query /
     gt_atoms | criteria / cut_idx);
  3. filters by gate verdict: by default only status=built; flagged items stay out;
  4. rebuilds contexts.json / cutoffs.json from messages.jsonl.

Usage:
    python3 tools/build_eval_bundle.py                         # all groups -> eval_bundles/bundle
    python3 tools/build_eval_bundle.py --out eval_bundles/x    # output directory
    python3 tools/build_eval_bundle.py --groups demo,1234      # selected groups only
    python3 tools/build_eval_bundle.py --include-flagged       # include flagged items (off by default)
    python3 tools/build_eval_bundle.py --layer L1              # one level only
    python3 tools/build_eval_bundle.py --include-legacy        # include runs without refreshed_by (not for release)
    python3 tools/build_eval_bundle.py --runs-dir /path/to/runs   # explicit runs directory

Outputs (eval_bundles/<out>/):
    items_adapted.jsonl  instances in harness form
    contexts.json        {group: [[turn, who, text], ...]}
    cutoffs.json         {iid: cut_turn}
    profiles.json        {group: [{member, slot, fact, turn}, ...]}
    manifest.json        ids bucketed by group / level / difficulty, for sampled runs
    bundle_meta.json     settings and statistics of this build
A harness needs only items_adapted.jsonl, contexts.json and cutoffs.json;
manifest.json is for stratified sampling.

run 目录 → 评测包: 构造端与评测端的唯一接口。自动发现群、映射字段、只收 built 题、重建上下文与截断点。
"""
import collections, json, os, re, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The runs directory can be set explicitly (--runs-dir / BUNDLE_RUNS_DIR) so that the
# source of a bundle is never implicit.
# run 目录可显式指定(--runs-dir / BUNDLE_RUNS_DIR), 避免来源不明。
_RUNS_DEFAULT = os.path.join(ROOT, 'pipeline_runs')


def _resolve_runs_dir():
    a = sys.argv
    if '--runs-dir' in a:
        return os.path.abspath(a[a.index('--runs-dir') + 1])
    if os.environ.get('BUNDLE_RUNS_DIR'):
        return os.path.abspath(os.environ['BUNDLE_RUNS_DIR'])
    return _RUNS_DEFAULT


RUNS_DIR = _resolve_runs_dir()
# Per-message character limit in contexts. 2000 keeps every evidence span of the
# released set inside the context; truncation of very long histories is left to the
# harness's own context budget.
# 上下文中每条消息的字数上限; 2000 可保证发布集中所有证据片段都在上下文内。
MSG_CHARS = int(os.environ.get('BUNDLE_MSG_CHARS', '2000'))


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def discover(include_legacy=False):
    """Discover group runs produced by the current pipeline.
    All three are required: items.jsonl, messages.jsonl, and pipeline.refreshed_by in
    run_meta.json. refreshed_by is written by refresh_run_meta.py, so only directories
    that went through the current pipeline have it; other directories under
    pipeline_runs/ are skipped. --include-legacy lifts this check (for comparison only,
    not for release).
    发现由当前 pipeline 生成的 run 目录: 三个条件缺一不可, refreshed_by 由 refresh_run_meta.py 写入。
    """
    out, skipped = [], []
    for name in sorted(os.listdir(RUNS_DIR)):
        d = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(d):
            continue
        if not (os.path.exists(os.path.join(d, 'items.jsonl'))
                and os.path.exists(os.path.join(d, 'messages.jsonl'))):
            continue
        meta_p = os.path.join(d, 'run_meta.json')
        cur = False
        if os.path.exists(meta_p):
            try:
                cur = bool(json.load(open(meta_p, encoding='utf-8'))
                           .get('pipeline', {}).get('refreshed_by'))
            except (ValueError, OSError):
                cur = False
        if cur or include_legacy:
            out.append((name.split('_')[0], d))
        else:
            skipped.append(name)
    if skipped:
        print(f'  [跳过历史产物 {len(skipped)} 个] {", ".join(skipped)}')
    return out


def dedup_query_lines(q):
    """Remove duplicate query lines. Grouping same-speaker blocks in P2 can join a message,
    its re-sent corrected version and an apology into one query. Only near-duplicate
    adjacent lines and pure apology lines are removed; the meaning is unchanged.
    题面按行去重: 只删近重复的相邻行与纯道歉行, 不改语义。
    """
    import difflib
    lines = [x.strip() for x in (q or '').split('\n') if x.strip()]
    out = []
    for ln in lines:
        if re.fullmatch(r'@?\s*bot\s*[，,]?\s*(抱歉|不好意思)[，,]?\s*(打错字了?|说错了?|口误)[。.！!]?', ln):
            continue  # pure apology line
        if any(difflib.SequenceMatcher(None, ln, p).ratio() > 0.85 for p in out):
            continue  # near-duplicate of an earlier line
        out.append(ln)
    return '\n'.join(out) if out else (q or '')


def mem_span(item, full):
    """Memory span = cut_turn - max(anchor.turn). Reports should be stratified by it: when
    the answer sits one or two turns before the question, a model without memory and one
    with the full history score alike, which dilutes the overall memory gain.
    记忆跨度 = 截断点 − 最晚锚点轮次; 评测报表应按它分层。
    """
    a = full.get('anchors') or []
    if not a or not isinstance(item.get('cut_idx'), int):
        return None
    return item['cut_idx'] - max(x['turn'] for x in a)


def main():
    out_dir = arg('--out', os.path.join(ROOT, 'eval_bundles', 'bundle'))
    want = set(filter(None, (arg('--groups', '') or '').split(',')))
    # --dirs names run directories explicitly instead of discovering them, e.g. when one
    # group has several run directories. The gid is taken from the basename, so
    # subdirectory paths work as well.
    # --dirs 显式指定 run 目录(同一群有多个 run 时使用); gid 取自目录名的最后一级。
    dirs = list(filter(None, (arg('--dirs', '') or '').split(',')))
    only_layer = arg('--layer')
    keep_flagged = '--include-flagged' in sys.argv
    os.makedirs(out_dir, exist_ok=True)
    # Check the runs directory up front, so a wrong path is reported clearly instead of as a
    # missing file deep inside one group.
    # 先校验 run 目录是否存在, 让路径错误一目了然。
    print(f'产线目录: {RUNS_DIR}')
    if not os.path.isdir(RUNS_DIR):
        sys.exit('✗ runs directory not found; pass it with --runs-dir / 产线目录不存在, 请用 --runs-dir 指定')
    print(f'输出目录: {out_dir}')

    items, ctx, cuts = [], {}, {}
    profs = {}
                   # Member profiles are added to the bundle: some anchors cite profile entries rather than
                   # messages, so a harness that shows only messages would hide them. Fields match the
                   # profile block of build_ctx (member/slot/fact/turn); the harness filters turn < cut per item.
                   # 评测包附带群档案: 部分锚点的证据在档案里; 按截断点裁剪由评测端逐题完成。
    stats = collections.Counter()
    per_group = {}
    src = ([(os.path.basename(n).split('_')[0], os.path.join(RUNS_DIR, n))
             for n in dirs] if dirs
           else discover('--include-legacy' in sys.argv))
    for gid, d in src:
        # The gid comes from the gid field of items.jsonl; the directory name is only a fallback,
        # because a directory such as demo_4368 does not start with the gid.
        # gid 以 items.jsonl 中的字段为准, 目录名只作兜底。
        rows_all = jl(os.path.join(d, 'items.jsonl'))
        gid_field = next((str(r['gid']) for r in rows_all if r.get('gid')), None)
        if gid_field and gid_field != gid:
            print(f'  [gid 以产物为准] 目录 {os.path.basename(d)} → 目录名解析 {gid!r}, '
                  f'items.gid {gid_field!r}, 采用后者')
            gid = gid_field
        if want and gid not in want:
            continue
        msgs = jl(os.path.join(d, 'messages.jsonl'))
        ctx[gid] = [[m['turn'], m.get('who'), (m.get('text') or '')[:MSG_CHARS]]
                    for m in sorted(msgs, key=lambda x: x['turn'])]
        pp = os.path.join(d, 'profile.json')
        if os.path.exists(pp):
            profs[gid] = [{'member': e.get('member'), 'slot': e.get('slot'),
                           'fact': e.get('fact'), 'turn': e.get('turn')}
                          for e in json.load(open(pp, encoding='utf-8'))]
        kept = collections.Counter()
        for it in rows_all:
            stats['total'] += 1
            if it.get('status') != 'built' and not keep_flagged:
                stats['skip_flagged'] += 1
                continue
            if only_layer and it['layer'] != only_layer:
                continue
            # The harness uses `kind` for L1/L2 and `cut_idx` for the cutoff.
            q = dedup_query_lines(it['query'])  # remove duplicate query lines
            row = {'iid': it['iid'], 'kind': it['layer'], 'group': gid,
                   'query': q, 'query_raw': it['query'] if q != it['query'] else None,
                   'query_v1': it.get('query_prev'),
                   'difficulty': it['difficulty'], 'source': 'pipeline',
                   'track': it.get('track') or 'jsonl',
                   'subtype': 'authored' if it.get('query_origin') == 'authored' else 'native',
                   'cut_idx': it['cut_turn'],
                   # Not read by the harness, but used for stratified reports
                   'op_type': it.get('op_type'), 'conflict_kind': it.get('conflict_kind'),
                   'query_origin': it.get('query_origin'),
                   'mem_span': mem_span({'cut_idx': it['cut_turn']}, it)}  # memory span
            # Revised-D3 marker. Arbitration-type and legacy update-type D3 items are also D3, so
            # downstream code that buckets only by difficulty would mix generations; this field
            # lets them be reported separately. It is written only on revised items, as in the
            # released dataset (writing null elsewhere would make a rebuilt bundle differ from the
            # release). Downstream code should test r.get('d3v2').
            # 新版 D3 标记, 只写在新版 D3 题上, 与发布数据一致; 下游用 r.get('d3v2') 判断。
            if it.get('d3v2'):
                row['d3v2'] = it['d3v2']
            if it['layer'] == 'L1':
                if not (it.get('gt_atoms') or []):
                    stats['skip_no_gt'] += 1
                    continue
                row.update(gt=it.get('gt'), gt_atoms=it['gt_atoms'],
                           distractor_atoms=it.get('distractor_atoms') or [])
            else:
                if not (it.get('criteria') or []):
                    stats['skip_no_criteria'] += 1
                    continue
                row['criteria'] = it['criteria']
            items.append(row)
            cuts[it['iid']] = it['cut_turn']
            kept[it['layer']] += 1
            kept[it['difficulty']] += 1
            stats['kept'] += 1
        per_group[gid] = dict(kept)

    with open(os.path.join(out_dir, 'items_adapted.jsonl'), 'w', encoding='utf-8') as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    json.dump(ctx, open(os.path.join(out_dir, 'contexts.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    json.dump(profs, open(os.path.join(out_dir, 'profiles.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    json.dump(cuts, open(os.path.join(out_dir, 'cutoffs.json'), 'w', encoding='utf-8'))

    # manifest: ids bucketed for stratified sampling (harness --manifest)
    man = collections.defaultdict(list)
    for r in items:
        man[f"{r['group']}/{r['kind']}/{r['difficulty']}"].append(r['iid'])
    json.dump(man, open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    # manifest_d3v2.json: revised D3 items only, bucketed like manifest.json, so they can
    # be evaluated on their own.
    # manifest_d3v2.json: 只含新版 D3 题, 分桶方式同 manifest.json, 可单独评测。
    man_v2 = collections.defaultdict(list)
    for r in items:
        if r.get('d3v2'):
            man_v2[f"{r['group']}/{r['kind']}/{r['difficulty']}"].append(r['iid'])
    n_v2 = sum(len(v) for v in man_v2.values())
    if n_v2:
        json.dump(man_v2, open(os.path.join(out_dir, 'manifest_d3v2.json'), 'w',
                               encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'  manifest_d3v2.json: {n_v2} 题 · {len(man_v2)} 桶 '
              f'(新版 D3, 可用 --manifest d3v2 单独跑)')

    meta = {'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'builder': 'tools/build_eval_bundle.py',
            'source': 'pipeline_runs/*/items.jsonl',
            'filter': {'status': 'built' if not keep_flagged else 'all',
                       'layer': only_layer or 'all', 'groups': sorted(want) or 'all'},
            'n_items': len(items), 'per_group': per_group, 'skipped': dict(stats),
            'layer': dict(collections.Counter(r['kind'] for r in items)),
            'difficulty': dict(collections.Counter(r['difficulty'] for r in items)),
            'track': dict(collections.Counter(r['track'] for r in items)),
            'query_origin': dict(collections.Counter(r['query_origin'] for r in items)),
            'mem_span': {'≤2': sum(1 for r in items if isinstance(r.get('mem_span'), int)
                                   and r['mem_span'] <= 2),
                         '3-20': sum(1 for r in items if isinstance(r.get('mem_span'), int)
                                     and 3 <= r['mem_span'] <= 20),
                         '>20': sum(1 for r in items if isinstance(r.get('mem_span'), int)
                                    and r['mem_span'] > 20),
                         'unknown': sum(1 for r in items if r.get('mem_span') is None)},
            'query_deduped': sum(1 for r in items if r.get('query_raw'))}
    json.dump(meta, open(os.path.join(out_dir, 'bundle_meta.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print(f"→ {out_dir}")
    print(f"  题 {len(items)} · 层 {meta['layer']} · 难度 {meta['difficulty']}")
    print(f"  轨 {meta['track']} · 来源 {meta['query_origin']}")
    print(f"  跨度 {meta['mem_span']} · 题面去重 {meta['query_deduped']} 道")
    print(f"  按群 {per_group}")
    print(f"  跳过 {dict(stats)}")
    tot = sum(len(t) for g in ctx.values() for _, _, t in g)
    print(f"  上下文 {len(ctx)} 群 · {sum(len(v) for v in ctx.values())} 条消息 · {tot:,} 字")


if __name__ == '__main__':
    main()
