#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 D3 material (multi-party conflicts): find windows where members hold mutually
exclusive positions on one decision and the conversation reaches an explicit outcome.

The model reads each window; validation stays in code.

What counts as a conflict (the prompt spells out four kinds of negatives):
  conflict = at least two human members take mutually exclusive positions on the same
             decision, resource or arrangement (they cannot both be satisfied).
  negatives: (1) differences of opinion that can coexist; (2) emotional friction with no
             concrete object; (3) one person changing their own mind (that is an update,
             handled by p4_d3); (4) quoted or template text (long bot replies, case
             studies, pasted prompt templates).
  resolved = the group reaches a visible outcome: one side gives way, a compromise is
             accepted, or someone decides. It must be cited with turn + span.

Checks in code (any failure -> reject, written to *_rejected for audit):
  - each side's turn exists, its span is a contiguous substring of that turn (after
    normalisation), and its member is the speaker of that turn and not a bot
  - the two sides are different members; the resolution turn comes after every
    position turn and its span verifies
Outputs: d3_conflicts.jsonl (counterpart of d3_updates.jsonl, consumed by P5) and
d3_conflicts_rejected.jsonl.
Usage: python3 pipeline/p4_conflict.py <gid> [--full] [--limit N] [--workers N] [--dry-run]
  default: targeted mode, only windows around keyword hits and arbitration-queue records;
  --full : sliding windows over the whole timeline (D3_WIN / D3_STRIDE, as in p4_d3).

P4 多方冲突抽取: 模型读窗口判断是否有成员在同一事项上持互斥立场并得到明确结果; 校验在代码中完成。
"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.llm import (config_banner, MODEL_MAIN, append_jsonl, chat_json, load_jsonl,
                          span_in, workers_arg, ParseFail)
from pipeline.p2_locate import rundir, load_msgs
from pipeline.p4_classify import ARBIT_PAT

CF_WIN_BACK = int(os.environ.get('CF_WIN_BACK', '30'))  # targeted window: turns before a hit
CF_WIN_FWD = int(os.environ.get('CF_WIN_FWD', '20'))  # turns after a hit (the resolution usually follows)
CF_MERGE_GAP = int(os.environ.get('CF_MERGE_GAP', '30'))  # merge hits closer than this into one window
CF_WIN = int(os.environ.get('D3_WIN', '40'))  # sliding-window size for --full
CF_STRIDE = int(os.environ.get('D3_STRIDE', '25'))
HUMAN_CHARS, BOT_CHARS = 200, 80

SYS = """你是数据标注流水线中的「多方冲突」抽取模块。给定一段群聊窗口, 找出其中**两个或以上人类成员在同一决策/资源/安排上持相互排斥立场**的真实冲突, 并判断群内是否出现了消解结果。输出会被程序逐字段校验: 每个 span 必须是所指 turn 消息原文的连续子串, member 必须与该 turn 的发言人一致, 校验失败该条即被丢弃。

# 什么算真冲突
- 立场相互排斥: 两方主张不能同时满足(钱怎么分摊、日期定哪天、方案选谁的、谁来承担)
- 立场发出者必须是**人类成员本人**(以该 turn 的发言人为准)

# 四类负例, 一律不输出
① 观点差异: 各说各话但不排他, 可以并存
② 情绪摩擦: 有火气没有具体争议标的
③ 单人自我改口: 同一个人先后说法不同(那是「更新」, 不是冲突)
④ 转述/模板文本: 消息里粘贴的案例、教学材料、提示词模板、bot 长文中描述的纠纷 —— 纠纷双方不是群成员本人

# 消解(resolved)
群内明确出现结果才算: 某方让步 / 折中方案被接受 / 有人拍板定案。resolution 给结果所在 turn 与原文 span。
只有争执没有结果 → resolved=false, resolution=null。

# 输出严格 JSON(没有冲突输出空数组; 宁缺毋滥)
{"conflicts":[{"topic":"争的是什么(≤20字)",
  "parties":[{"member":"发言人名","stance":"其立场(≤25字)","turn":123,"span":"该turn原文连续子串(≤25字)"}],
  "resolved":true,"resolution":{"turn":130,"outcome":"结果(≤25字)","span":"原文连续子串(≤25字)"}}]}"""


def windows_targeted(msgs, d):
    hits = set()
    for m in msgs:
        if m.get('kind') != 'bot' and ARBIT_PAT.search(m.get('text') or ''):
            hits.add(m['turn'])
    for r in load_jsonl(os.path.join(d, 'arbitration_queue.jsonl')):
        if r.get('stage') == 'P4-3' and isinstance(r.get('turn'), int):
            hits.add(r['turn'])
    if not hits:
        return []
    wins, cur = [], None
    for t in sorted(hits):
        lo, hi = t - CF_WIN_BACK, t + CF_WIN_FWD
        if cur and lo <= cur[1] + CF_MERGE_GAP:
            cur[1] = hi
        else:
            cur = [lo, hi]
            wins.append(cur)
    return [(max(0, lo), hi) for lo, hi in wins]


def windows_full(msgs):
    last = max(m['turn'] for m in msgs)
    return [(s, min(s + CF_WIN, last)) for s in range(0, last, CF_STRIDE)]


def render(msgs, lo, hi):
    L = []
    for m in msgs:
        if not (lo <= m['turn'] <= hi):
            continue
        t = (m.get('text') or '').replace('\n', ' ')
        cap = BOT_CHARS if m.get('kind') == 'bot' else HUMAN_CHARS
        tag = '[bot]' if m.get('kind') == 'bot' else ''
        L.append(f"#{m['turn']} [{m.get('who')}]{tag} {t[:cap]}")
    return '\n'.join(L)


def verify(c, by_turn):
    """Literal checks in code; returns (ok, reason).
    代码侧字面校验, 返回 (是否通过, 原因)。
    """
    ps = c.get('parties') or []
    if len(ps) < 2:
        return False, 'parties<2'
    members = set()
    for p in ps:
        m = by_turn.get(p.get('turn'))
        if m is None:
            return False, f"turn_missing:{p.get('turn')}"
        if m.get('kind') == 'bot':
            return False, f"party_is_bot:{p.get('turn')}"
        if (p.get('member') or '') != (m.get('who') or ''):
            return False, f"member_mismatch:{p.get('turn')}"
        if not span_in(p.get('span'), m.get('text')):
            return False, f"span_not_found:{p.get('turn')}"
        members.add(p['member'])
    if len(members) < 2:
        return False, 'same_member(自我改口应属update)'
    if c.get('resolved'):
        r = c.get('resolution') or {}
        m = by_turn.get(r.get('turn'))
        if m is None or not span_in(r.get('span'), m.get('text')):
            return False, f"resolution_unverified:{r.get('turn')}"
        if r['turn'] <= max(p['turn'] for p in ps):
            return False, 'resolution_before_stances'
    return True, ''


def main():
    args = sys.argv[1:]
    gid = next((a for a in args if not a.startswith('-')), 'demo')
    full = '--full' in args
    limit = int(args[args.index('--limit') + 1]) if '--limit' in args else 0
    dry = '--dry-run' in args
    workers = workers_arg(args, default=1)
    d = rundir(gid)
    msgs = load_msgs(d)
    by_turn = {m['turn']: m for m in msgs}
    wins = windows_full(msgs) if full else windows_targeted(msgs, d)
    out_p = os.path.join(d, 'd3_conflicts.jsonl')
    rej_p = os.path.join(d, 'd3_conflicts_rejected.jsonl')
    done = {tuple(r.get('window') or ()) for r in load_jsonl(out_p)} \
        | {tuple(r.get('window') or ()) for r in load_jsonl(rej_p) if r.get('scan_marker')}
    todo = [w for w in wins if tuple(w) not in done]
    if limit:
        todo = todo[:limit]
    print(config_banner(), flush=True)
    print(f"P4-conflict[{'full' if full else 'targeted'}]: 窗口 {len(wins)} · 已扫 {len(wins)-len(todo)} "
          f"· 本轮 {len(todo)}", flush=True)
    if dry:
        for lo, hi in todo:
            print(f'  [{lo},{hi}] {sum(1 for m in msgs if lo<=m["turn"]<=hi)} 条消息')
        return

    def scan(w):
        lo, hi = w
        try:
            v = chat_json(d, 'p4_conflict', SYS, render(msgs, lo, hi), model=MODEL_MAIN,
                          max_tokens=800)
        except ParseFail as e:
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P4-CF', 'window': [lo, hi], 'err': str(e)})
            return
        except Exception as e:                                   # noqa: BLE001
            # A permanent error on one window (e.g. a content-policy refusal) is recorded instead
            # of aborting the stage; the scan_marker lets a resumed run skip this window.
            # 单个窗口的永久性错误只记录不中断; 写 scan_marker 让续跑跳过该窗口。
            append_jsonl(os.path.join(d, 'd3_conflicts_rejected.jsonl'),
                         {'window': [lo, hi], 'scan_marker': True,
                          'reject': f'api_error:{type(e).__name__}'})
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P4-CF', 'window': [lo, hi], 'err': f'{type(e).__name__}: {e}'[:500]})
            print(f'  ⚠ [{lo},{hi}] {type(e).__name__} → 跳过并留痕', flush=True)
            return
        found = v.get('conflicts') or []
        for c in found:
            ok, why = verify(c, by_turn)
            row = {'window': [lo, hi], 'topic': c.get('topic'), 'parties': c.get('parties'),
                   'resolved': bool(c.get('resolved')), 'resolution': c.get('resolution'),
                   'by': f'model:conflict_scan({MODEL_MAIN})'}
            if ok:
                append_jsonl(out_p, row)
                r = c.get('resolution') or {}
                print(f"  ✓ [{lo},{hi}] {c.get('topic')} · {len(c['parties'])}方 "
                      f"resolved={row['resolved']}" + (f" @#{r.get('turn')}" if row['resolved'] else ''),
                      flush=True)
            else:
                row['reject'] = why
                append_jsonl(rej_p, row)
                print(f"  ✗ [{lo},{hi}] {c.get('topic')} reject={why}", flush=True)
        if not found:
            # Empty windows are recorded too (scan_marker); resuming and recall audits need them.
            # 空窗口也留记录, 供断点续跑与召回审计使用。
            append_jsonl(rej_p, {'window': [lo, hi], 'scan_marker': True, 'reject': 'no_conflict'})

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(scan, todo))
    rows = load_jsonl(out_p)
    print(f'\n汇总: 冲突对 {len(rows)} (resolved {sum(1 for r in rows if r["resolved"])}) → {out_p}')


if __name__ == '__main__':
    main()
