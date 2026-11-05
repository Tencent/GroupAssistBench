#!/usr/bin/env python3
"""Conflict-pair screening: every update event found by p4_d3 is judged three times
independently by the model, labelled red / yellow / green, and written back to the
run directory together with a review sheet for humans.

Why: D3 instances are built only from conflict pairs and never repair them, so a pair
that is not a real change of mind must be filtered out before building.
Labels: each vote picks one of five classes (true_update / different_slot /
range_extension / goal_vs_status / rephrase), judged from the pair and the turns
around it, so pairs can be screened before any instance exists.
  green  : all votes say true_update
  yellow : votes split
  red    : the majority says it is a pseudo-update (or all three votes differ)
Only green pairs, or red/yellow pairs a human confirms as true_update, are built.

Outputs (under pipeline_runs/<run>/):
  r7_pairs_votes.jsonl   individual votes (resumable)
  d3_pairs_r7.json       per-pair verdict {pair_key: {auto, majority, votes, human, ...}},
                         read by p5_build.py --rt2-d3
  r7_pairs_review.md     review sheet, red -> yellow -> green, with the two source turns

Usage (package root):
  python3 pipeline/r7_pairs_screen.py <run>... [--rounds 3] [--workers 3] [--review-md PATH]
  python3 pipeline/r7_pairs_screen.py <run>... --tally-only        # recompute verdicts and sheet, no model calls
  python3 pipeline/r7_pairs_screen.py <run> --human <verdicts.jsonl>
      # merge human verdicts, one per line: {"pair_key":"165->178","verdict_human":"true_update","note":""}

冲突对筛查: 每个改口事件独立投票三次 → 红/黄/绿 → 写回 run 目录并生成人判表。
只有绿色, 或经人工确认为 true_update 的红/黄事件, 才会用于出 D3 题。
"""
import argparse, json, os, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.llm import chat_json, append_jsonl, load_jsonl, ParseFail, MODEL_MAIN   # noqa: E402
from pipeline.p5_build import load_d3_pairs, d3v2_dedupe_pairs                       # noqa: E402

RUNS = os.path.join(ROOT, 'pipeline_runs')
# Votes and verdicts are written into the same run directory, so their model calls are
# counted in that run's usage.jsonl and run_meta (stage 'r7_pairs_vote').
# 投票与结论写回同一个 run 目录, 筛查的调用计入该 run 的 usage 与 run_meta。
OUT = RUNS
FAKE = ('different_slot', 'range_extension', 'goal_vs_status', 'rephrase')
CTX = 3  # turns of context taken on each side of both turns in a pair

SYS = """你是数据集质检的「冲突更新真实性」评审模块。给定群聊里一次被自动识别的「改口事件」(旧值 → 新值)以及两端的原文与前后几轮, 判断旧值与新值是否构成**同一事实槽位的真实取代**。只输出 JSON。
  true_update      真取代: 同一件事、同一主体, 后者替代了前者
  different_slot   不同维度或不同主体(体脂率 vs 体型描述; A 的饮品 vs B 的饮品)
  range_extension  范围扩充而非取代(过敏清单加长; 只做徽章 → 徽章+明信片属于扩充还是取代按语义判)
  goal_vs_status   目标 vs 现状(想减到 60kg vs 现在 65kg)
  rephrase         同一状态换种说法/称呼并存(「堆成鬼样」→「全倒床上」)
注意: 只看事实是否被取代, 不看措辞相似度; 「@某人 那我就按你说的」这类回应, 主体是被回应的人, 不是所有人。
输出严格 JSON: {"verdict":"true_update|different_slot|range_extension|goal_vs_status|rephrase","reason":"≤30字"}"""


def pair_key(p):
    return f"{p['old_anchor_turn']}->{p['update_turn']}"


def build_user(p, by_turn):
    def win(t):
        rows = []
        for k in range(t - CTX, t + CTX + 1):
            m = by_turn.get(k)
            if m:
                mark = ' ◀' if k == t else ''
                rows.append(f"#{k} [{m.get('who')}] {(m.get('text') or '')[:160]}{mark}")
        return '\n'.join(rows)
    return (f"# 改口事件\n事项: {p.get('slot')}\n旧值: {p.get('old_value')}  (锚点 #{p['old_anchor_turn']}: {p.get('old_anchor_fact')})\n"
            f"新值: {p.get('new_value')}  (更新轮 #{p['update_turn']})\n识别依据: {p.get('why') or ''}\n\n"
            f"# 旧值所在原文(◀ 为该轮)\n{win(p['old_anchor_turn'])}\n\n# 新值所在原文(◀ 为该轮)\n{win(p['update_turn'])}")


def load_run(name):
    d = os.path.join(RUNS, name)
    pairs = d3v2_dedupe_pairs(load_d3_pairs(d))
    by_turn = {m['turn']: m for m in load_jsonl(os.path.join(d, 'messages.jsonl'))}
    return d, pairs, by_turn


def prior_verdicts(d):
    """Pairs that already have a verdict in d3_pairs_r7.json. They are not voted again;
    their stored verdict, which includes the individual votes, is reused as is.
    d3_pairs_r7.json 中已有结论的冲突对: 不再重投, 直接沿用。
    """
    vp = os.path.join(d, 'd3_pairs_r7.json')
    prev = json.load(open(vp, encoding='utf-8')) if os.path.exists(vp) else {}
    return {k: v for k, v in prev.items() if v.get('votes') and v.get('auto') not in (None, 'unvoted')}


def vote(name, rounds, workers):
    d, pairs, by_turn = load_run(name)
    od = os.path.join(OUT, name); os.makedirs(od, exist_ok=True)
    res_p = os.path.join(od, 'r7_pairs_votes.jsonl')
    done = {(r['pair_key'], r['round']) for r in load_jsonl(res_p)} if os.path.exists(res_p) else set()
    reuse = prior_verdicts(d)
    jobs = [(p, rd) for p in pairs for rd in range(1, rounds + 1)
            if (pair_key(p), rd) not in done and pair_key(p) not in reuse]
    print(f'[{name}] 冲突对 {len(pairs)} · 复用已有结论 {sum(1 for p in pairs if pair_key(p) in reuse)} '
          f'· 待投 {len(jobs)} 调用 · model={MODEL_MAIN}', flush=True)

    def one(job):
        p, rd = job
        rec = {'pair_key': pair_key(p), 'round': rd, 'by': f'model:{MODEL_MAIN}'}
        try:
            v = chat_json(od, 'r7_pairs_vote', SYS, build_user(p, by_turn), max_tokens=200)
            rec['verdict'] = v.get('verdict'); rec['reason'] = str(v.get('reason', ''))[:60]
        except ParseFail as e:
            rec['error'] = str(e)[:120]
        append_jsonl(res_p, rec)
        return rec
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, jobs))


def tally(name, human_rows=None):
    d, pairs, by_turn = load_run(name)
    res_p = os.path.join(OUT, name, 'r7_pairs_votes.jsonl')
    votes = {}
    for r in (load_jsonl(res_p) if os.path.exists(res_p) else []):
        if r.get('verdict'):
            votes.setdefault(r['pair_key'], {})[r['round']] = (r['verdict'], r.get('reason', ''))
    vp = os.path.join(d, 'd3_pairs_r7.json')
    prev = json.load(open(vp, encoding='utf-8')) if os.path.exists(vp) else {}
    reuse = prior_verdicts(d)
    out = {}
    for p in pairs:
        k = pair_key(p)
        if k in reuse and not votes.get(k):  # reuse: no new votes, keep the stored verdict
            out[k] = dict(reuse[k])
            continue
        vs = [votes.get(k, {})[r] for r in sorted(votes.get(k, {}))]
        cnt = Counter(v for v, _ in vs)
        top = cnt.most_common(1)
        majority = top[0][0] if top and top[0][1] >= 2 else ('split' if vs else None)
        unanimous = len(cnt) == 1 and sum(cnt.values()) >= 3
        auto = ('red' if majority in FAKE or majority == 'split' else 'yellow' if not unanimous else 'green') if vs else 'unvoted'
        rec = {'auto': auto, 'majority': majority, 'votes': [{'verdict': v, 'reason': r} for v, r in vs],
               'slot': p.get('slot'), 'old_value': p.get('old_value'), 'new_value': p.get('new_value'),
               'old_anchor_turn': p['old_anchor_turn'], 'update_turn': p['update_turn'],
               'human': (prev.get(k) or {}).get('human'), 'human_note': (prev.get(k) or {}).get('human_note'),
               'tallied_at': time.strftime('%Y-%m-%d %H:%M')}
        out[k] = rec
    for h in (human_rows or []):
        if h.get('pair_key') in out:
            out[h['pair_key']]['human'] = h.get('verdict_human'); out[h['pair_key']]['human_note'] = h.get('note')
    json.dump(out, open(vp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    c = Counter(v['auto'] for v in out.values())
    print(f"[{name}] 红 {c['red']} · 黄 {c['yellow']} · 绿 {c['green']} · 未投 {c['unvoted']} · 已人判 {sum(1 for v in out.values() if v['human'])} → {vp}")
    return out, by_turn


def review_md(path, results):
    order = {'red': 0, 'yellow': 1, 'green': 2, 'unvoted': 3}
    tag = {'red': '🔴红', 'yellow': '🟡黄', 'green': '🟢绿', 'unvoted': '⚪未投'}
    rows = [(name, k, v, by_turn) for name, (res, by_turn) in results.items() for k, v in res.items()]
    rows.sort(key=lambda x: (order[x[2]['auto']], x[0], x[2]['update_turn']))
    tot = Counter(v['auto'] for _, _, v, _ in rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# 冲突对人判表（{time.strftime('%Y-%m-%d')}）\n\n")
        f.write(f"红 **{tot['red']}** · 黄 **{tot['yellow']}** · 绿 **{tot['green']}**。从上往下判; 绿的抽查。"
                "判完把结论写成 jsonl（每行 `{\"pair_key\":\"165->178\",\"verdict_human\":\"true_update\",\"note\":\"\"}`），"
                "用 `--human 文件` 合并。人判 true_update 的红/黄对才会出题。\n\n")
        for name, k, v, by_turn in rows:
            f.write(f"## {tag[v['auto']]} {name} · {k} — 事项「{v['slot']}」: `{v['old_value']}` → `{v['new_value']}`  (模型多数: {v['majority']})\n\n")
            for i, x in enumerate(v['votes'], 1):
                f.write(f"- r{i}: {x['verdict']} — {x['reason']}\n")
            f.write('\n| | turn | 发言人 | 原文 |\n|---|---|---|---|\n')
            for role, t in (('旧', v['old_anchor_turn']), ('新', v['update_turn'])):
                m = by_turn.get(t) or {}
                txt = (m.get('text') or '').replace('|', '｜').replace('\n', ' ')[:200]
                f.write(f"| {role} | {t} | {m.get('who')} | {txt} |\n")
            f.write('\n**人判**: ☐ true_update ☐ different_slot ☐ range_extension ☐ goal_vs_status ☐ rephrase'
                    + (f"  （已判: {v['human']}）" if v.get('human') else '') + '\n\n---\n\n')
    print(f'→ {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+'); ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=3); ap.add_argument('--tally-only', action='store_true')
    ap.add_argument('--human'); ap.add_argument('--review-md')
    a = ap.parse_args()
    human = [json.loads(l) for l in open(a.human, encoding='utf-8')] if a.human else None
    results = {}
    for name in a.runs:
        if not a.tally_only and not a.human:
            vote(name, a.rounds, a.workers)
        results[name] = tally(name, human)
    path = a.review_md or (os.path.join(OUT, a.runs[0], 'r7_pairs_review.md') if len(a.runs) == 1
                           else os.path.join(OUT, f"r7_pairs_review_{time.strftime('%Y%m%d')}.md"))
    review_md(path, results)


if __name__ == '__main__':
    main()
