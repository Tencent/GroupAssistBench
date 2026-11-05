#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 D3 material (update-overwrite): find positions where an established fact is changed
to a new value. The only D3 extraction step that calls a model.

Why not rules alone: an update's old and new value need not be numbers or share any
wording ("leaning towards blue-green" -> "switch to a mint-blue / sunset-orange
contrast"), so lexical pairing of anchors and update sentences has very low precision.

Why windows rather than single messages: in group chat an update is often reached over
several turns (a proposal, a revised draft, then "yes, that's it, let's finalise"); the
confirming turn may contain no update wording at all, so judging single messages misses it.

Method:
  1. Code cuts the timeline into overlapping windows (bot turns are included as context
     but are never the subject of a change).
  2. For each window the model is asked: was a previously established fact changed to a
     new value, and did the change take effect or get confirmed? If so it returns
     {slot, old_value, new_value, decided_turn}.
  3. Code matches slot / old_value to an anchor BEFORE decided_turn; only a match counts
     as D3 evidence.
The model only reads what is being changed; matching and assignment stay in code, and
turn order prevents any future leakage.

Output: d3_updates.jsonl (merged into detect_update by p4_classify when present)
Usage: python3 pipeline/p4_d3.py <gid> [--dry-run] [--rebuild]

P4 D3 更新事件抽取: 按重叠窗口让模型找「已确立的事实被改为新值并生效」的事件,
再在代码中把旧值回匹到更新之前的锚点; 匹配上才算 D3 素材。
"""
import difflib, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.llm import MODEL_MAIN, append_jsonl, chat_json, load_jsonl, ParseFail
from pipeline.p2_locate import rundir, load_msgs

WIN = int(os.environ.get('D3_WIN', '30'))  # window length (turns)
STRIDE = int(os.environ.get('D3_STRIDE', '20'))  # stride; one-third overlap so that events are not cut in half
MSG_CHARS = int(os.environ.get('D3_MSG_CHARS', '120'))
MATCH_SIM = float(os.environ.get('D3_MATCH_SIM', '0.45'))  # minimum similarity for matching slot / old_value to an anchor

SYS = """你在一段群聊里找「槽位更新」：某个**此前已经确立**的事实，在这段对话里被改成了新值。

判是的条件（三条全满足）：
1. 旧值是此前群里已经确立的（不是第一次提出的新事物）；
2. 改的是**同一个槽位**（同一件事的取值），不是新增一件无关的事；
3. 变更**已经生效或被确认** —— 可以是一句话直接宣告（「我们把规则改为6公里了」），
   也可以是多轮达成（有人提议 → 执行 → 有人确认「就这样」/「定稿」）。
   只有提议、没人接、也没执行的，判否。

判否的典型：动词/成语里的「提前/变成/改到」（「提前做预案」「改稿改到崩溃」「变成我的锅」）；
对图片文案的一次性编辑请求；单纯的建议或询问且无下文；同一事实的复述。

decided_turn 填**变更生效/被确认的那一轮**（宣告型就填宣告那轮；协商型填确认那轮）。

输出严格 JSON（没有就 is_update=false，其余留空）：
{"is_update":true/false,"slot":"被改的是什么(≤12字)","old_value":"旧值(≤20字)","new_value":"新值(≤20字)","decided_turn":整数,"why":"≤24字"}"""


def _norm(s):
    return re.sub(r'[《》「」“”"\'（）()、，。！？；：\s]', '', s or '')


# "Value not yet decided" markers are searched anywhere in the string, not only at the start.
# 「尚未确定」类标记在整串中搜索, 不只匹配开头。
NULL_OLD = re.compile(r'未定|待定|不详|未确定|待比较|尚未|未提及|未知|未指明|未明确|'
                      r'未说明|没有明确|无明确|原先时间|讨论中|候选')

MATCH_SYS = """给你一次「槽位更新」事件和一批候选记忆锚点。选出**被这次变更覆盖掉的旧值锚点**。

要求：该锚点陈述的必须是**同一个槽位在变更前的取值**。同一件事换个说法算同一槽位
（例：锚点说「倾向用蓝绿色」、事件说「配色从经典冷色改成撞色」——都是这张海报的配色槽位）。
只是话题相关、但不是这个槽位的取值，不算。
**方向很重要**：要找的是**变更前**的取值。如果某条锚点陈述的已经是变更后的新值，那不是答案。
找不到就返回 -1 —— 宁可返回 -1，也不要选一条方向相反的。

输出严格 JSON：{"turn":锚点turn或-1,"why":"≤20字"}"""


def _contains_new(fact, new_value):
    """Is this anchor already stating the NEW value? The lexically closest anchor is often
    the one carrying the new value; picking it as the old value would move a correct fact
    into distractor_atoms, which is worse than missing the event, so it is rejected outright.
    该锚点是否已在陈述新值。选错方向会把正确事实放进干扰项, 比漏检更有害, 故直接拒绝。
    """
    nv = _norm(new_value)
    if len(nv) < 2:
        return False
    f = _norm(fact)
    if nv in f:
        return True
    grams = {nv[i:i + 2] for i in range(len(nv) - 1)}
    fg = {f[i:i + 2] for i in range(len(f) - 1)}
    return bool(grams) and len(grams & fg) / len(grams) >= 0.6


def _lex_shortlist(ev, inv, upd_turn, k=20):
    """Pre-filter candidate anchors by lexical similarity and pass only the top k to the model.
    All three direction checks live here: earlier in time, not a bot turn, not already
    stating the new value.
    先按词面相似度粗筛候选锚点, 只把前 k 个交给模型; 三道方向检查都在这里。
    """
    key = _norm((ev.get('old_value') or '') + (ev.get('slot') or ''))
    nv = ev.get('new_value') or ''
    scored = []
    for a in inv:
        if a['turn'] >= upd_turn:
            continue  # the old-value anchor must precede the update (no future leakage)
        if a.get('bot_sourced'):
            continue  # bot-generated content is not group memory and cannot be the old value
        if _contains_new(a.get('fact'), nv):
            continue  # direction check: this anchor already states the new value
        f = _norm(a.get('fact'))
        s = difflib.SequenceMatcher(None, key, f).ratio()
        if ev.get('old_value') and _norm(ev['old_value']) and _norm(ev['old_value']) in f:
            s = max(s, 0.9)
        scored.append((s, a))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def match_anchor(ev, inv, upd_turn, d=None):
    """Match an event to its old-value anchor in two steps:
      1. a lexically similar anchor (>= MATCH_SIM) is taken directly;
      2. otherwise the top 20 candidates go to the model, for updates whose old and new
         wording share almost nothing.
    Candidates are limited to anchors before the update, so the model cannot look ahead.
    两步回匹旧值锚点: 词面足够相似直接取; 否则把前 20 个候选交给模型选。候选只取更新之前的锚点。
    """
    if NULL_OLD.search((ev.get('old_value') or '').strip()):
        return None  # old value is "not yet decided": a first decision, not an update
    short = _lex_shortlist(ev, inv, upd_turn)
    if not short:
        return None
    if short[0][0] >= MATCH_SIM:
        return short[0][1], round(short[0][0], 3)
    if d is None:
        return None
    lst = '\n'.join(f"#{a['turn']} {a['fact'][:60]}" for _, a in short)
    try:
        v = chat_json(d, 'p4_d3match', MATCH_SYS,
                      f"事件: 槽位「{ev.get('slot')}」 {ev.get('old_value')} → {ev.get('new_value')}"
                      f"\n(生效于 #{upd_turn})\n\n候选锚点:\n{lst}", max_tokens=200)
    except ParseFail:
        return None
    t = v.get('turn')
    if not isinstance(t, int) or t < 0 or t >= upd_turn:
        return None
    hit = next((a for _, a in short if a['turn'] == t), None)
    if hit and _contains_new(hit.get('fact'), ev.get('new_value') or ''):
        return None  # the model may still pick a new-value anchor; check again on the way out
    return (hit, 'model') if hit else None


# Minimum span length for back-filling. Shorter old values cannot be located by literal
# match without hitting coincidental common words, so precision is preferred over recall.
# 补抽的最短片段长度。过短的旧值无法靠字面可靠定位, 宁可少补。
BACKFILL_MIN_SPAN = int(os.environ.get('D3_BACKFILL_MIN_SPAN', '4'))


def _lcs(a, b):
    """Longest common substring; returns (length, start in b). Used to locate the model's
    old_value summary as a contiguous span of the original message, because an
    evidence_span must appear verbatim in the turn it cites.
    最长公共子串, 用于把 old_value 定位回原文中真实存在的连续片段。
    """
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    best, pos = 0, 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > best:
                    best, pos = m[i][j], j - m[i][j]
    return best, pos


def backfill_anchor(ev, msgs, upd_turn):
    """Back-fill an old-value anchor from the transcript.
    A real update can lack an old-value anchor because P3 judged the original statement
    not factual enough to extract. Without it no D3 instance can be built, since the old
    value must become a distractor with a verifiable #turn citation. This locates
    old_value in the turns BEFORE the update and uses a contiguous substring that really
    exists there as the evidence span, so the usual literal checks still apply.
    Same three constraints as _lex_shortlist: earlier in time, human turns preferred, not
    already stating the new value.
    旧值锚点补抽: 在更新之前的原文中定位旧值, 取真实存在的连续片段作证据, 仍需通过字面校验。
    """
    ov = (ev.get('old_value') or '').strip()
    if NULL_OLD.search(ov):
        return None  # "not yet decided" is a first decision; there is no old value to back-fill
    if len(_norm(ov)) < BACKFILL_MIN_SPAN:
        return None
    nv = ev.get('new_value') or ''
    best = None
    for m in msgs:
        if m['turn'] >= upd_turn:
            continue  # no future leakage
        t = m.get('text') or ''
        if not t or _contains_new(t, nv):
            continue  # this message already states the new value
        n, pos = _lcs(ov, t)
        if n < BACKFILL_MIN_SPAN:
            continue
        span = t[pos:pos + n]
        # Human turns are preferred: they win ties; bot turns are still accepted but marked.
        # 人类消息优先: 同分时人类胜出; bot 消息仍可取但会标记。
        score = (n, 0 if m.get('kind') == 'bot' else 1, m['turn'])
        if best is None or score > best[0]:
            best = (score, {'turn': m['turn'], 'fact': f"{ev.get('slot')}此前为{ov}",
                            'type': '状态', 'by': 'rule:d3_backfill',
                            'evidence_span': span,
                            'bot_sourced': m.get('kind') == 'bot',
                            'used_by_queries': [], 'consumed_by_items': []})
    return best[1] if best else None


def main():
    gid = next((a for a in sys.argv[1:] if not a.startswith('-')), None)
    dry = '--dry-run' in sys.argv
    d = rundir(gid)
    msgs = load_msgs(d)
    inv = json.load(open(os.path.join(d, 'anchor_inventory.json'), encoding='utf-8'))
    turns = sorted(m['turn'] for m in msgs)
    M = {m['turn']: m for m in msgs}
    lo, hi = turns[0], turns[-1]
    wins = [(a, a + WIN) for a in range(lo, hi + 1, STRIDE)]
    print(f'P4-D3[{gid}]: 时间线 {lo}–{hi} · 窗口 {len(wins)} 个(长 {WIN} 步 {STRIDE})', flush=True)
    out_p = os.path.join(d, 'd3_updates.jsonl')
    lock = out_p + '.lock'
    if os.path.exists(lock):
        raise SystemExit(f'{lock} 存在: 另一个 p4_d3 正在写同一个群, 拒绝并发(会污染产物)')
    # Reuse: an existing d3_updates.jsonl is a finished result (the lock above is only released
    # on success) and is kept as is, like every other stage's output. Pass --rebuild to re-detect.
    # 复用: 已有 d3_updates.jsonl 即为完成态, 原样保留; 加 --rebuild 才重新检测。
    if os.path.exists(out_p) and '--rebuild' not in sys.argv and not dry:
        n = sum(1 for _ in load_jsonl(out_p))
        print(f'P4-D3[{gid}]: 复用已有 {out_p} ({n} 条更新事件); 重新检测请加 --rebuild', flush=True)
        return
    open(lock, 'w').close()
    if os.path.exists(out_p):
        os.remove(out_p)
    n_upd = n_hit = 0
    seen = set()
    backfilled = []
    for a, b in wins:
        body = '\n'.join(f"#{t} [{M[t]['who']}{'/bot' if M[t].get('kind') == 'bot' else ''}] "
                          f"{(M[t].get('text') or '')[:MSG_CHARS]}"
                          for t in turns if a <= t < b)
        if not body.strip():
            continue
        if dry:
            print(f'  窗口 {a}-{b}: {len(body)} 字')
            continue
        try:
            ev = chat_json(d, 'p4_d3', SYS, f'对话片段:\n{body}', max_tokens=400)
        except ParseFail as e:
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P4-D3llm', 'window': [a, b], 'err': str(e)})
            continue
        if not ev.get('is_update'):
            continue
        dt = ev.get('decided_turn')
        if not isinstance(dt, int) or not (a <= dt < b):
            dt = b - 1  # missing or out of range: fall back to the window's last turn
        # Deduplicate by (decided turn, new value): overlapping windows report the same event
        # more than once, each time with a differently worded slot name.
        # 按(生效轮次, 新值)去重: 重叠窗口会重复报告同一事件, 且每次 slot 名不同。
        key = (dt, _norm(ev.get('new_value'))[:12])
        if key in seen:
            continue  # overlapping windows report the same event more than once
        seen.add(key)
        n_upd += 1
        hit = match_anchor(ev, inv, dt, d=d)
        if hit is None:  # not in the inventory: back-fill from the transcript
            bf = backfill_anchor(ev, msgs, dt)
            if bf is not None:
                inv.append(bf)  # added to the inventory for p5_build's D3 path
                backfilled.append(bf)
                hit = (bf, 'backfill')
        rec = {'update_turn': dt, 'window': [a, b], 'slot': ev.get('slot'),
               'old_value': ev.get('old_value'), 'new_value': ev.get('new_value'),
               'why': ev.get('why'), 'by': f'model:slot_update({MODEL_MAIN})'}
        if hit:
            x, sc = hit
            rec.update(old_anchor_turn=x['turn'], old_anchor_fact=x['fact'], match_by=sc)
            n_hit += 1
        else:
            rec['reason'] = 'no_anchor_match(库存里找不到对应旧值锚, 不能定 D3)'
        append_jsonl(out_p, rec)
        print(f"  窗口{a}-{b} #{dt} 「{ev.get('slot')}」{ev.get('old_value')}→{ev.get('new_value')}"
              f"  {'✓ 锚点#' + str(hit[0]['turn']) + f'({hit[1]})' if hit else '✗ 无锚点'}", flush=True)
    if backfilled:  # write back-filled anchors to the inventory
        inv_p = os.path.join(d, 'anchor_inventory.json')
        cur = json.load(open(inv_p, encoding='utf-8'))
        have = {(a['turn'], a.get('fact')) for a in cur}
        add = [a for a in backfilled if (a['turn'], a['fact']) not in have]
        if add:
            json.dump(sorted(cur + add, key=lambda x: x['turn']),
                      open(inv_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.remove(lock)
    print(f'P4-D3[{gid}] 完成: 去重后更新事件 {n_upd} · 回匹到锚点 {n_hit}'
          f'(其中补抽 {len(backfilled)}) → {out_p}')


if __name__ == '__main__':
    main()
