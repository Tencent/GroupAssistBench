#!/usr/bin/env python3
# P4 Assign dependency type (paper 3.2 "assigning dependency types"): fold near-duplicate
# anchors into memory units, then derive D1/D2 (= paper M1/M2), operation type and event
# clustering. Fully deterministic, no model calls. Four independent fields:
#   difficulty     : D1 = at most one substantive memory unit, D2 = D2_MIN_UNITS or more
#                    (persona/background entries do not count) - rule:memory_units
#   D3-update      : one fact slot with two or more values, detected structurally
#                    (update wording + changed value) - rule:update_detect
#   D3-arbitration : multi-party disputes are only routed to arbitration_queue here,
#                    never assigned automatically
#   layer          : not decided here; existing human / inherited labels are passed
#                    through, everything else stays pending (see p4_layer.py)
#   op_type        : recall / aggregate / resolve - rule:op_derive (from anchor structure
#                    and conflict flags; P5 may override)
#   event clusters : union-find over shared anchor turns -> event_id
# Output: classify.jsonl
# Usage: python3 pipeline/p4_classify.py <gid> [--reset-layer]
#   --reset-layer  do not carry over layer values filled in a previous run (default: carry over)
# P4 判定(纯代码): 把近重复锚点折叠为记忆单元, 据此推出 D1/D2、操作类型与事件簇; layer 不在此判。
import difflib, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.llm import append_jsonl, load_jsonl
from pipeline.p2_locate import rundir, load_msgs

UPDATE_PAT = re.compile(r'改到|改成|改为|换成|变成|调整为|推迟|提前|更正|不是.{0,6}是|以.{1,8}为准')
# Only resource-allocation words; words for arguing or disagreeing are not listed,
# because they also appear in ordinary emotional or advice questions.
# 只保留资源分配类词; 「吵/争执」等词常见于情绪类提问, 不列入。
ARBIT_PAT = re.compile(r'分摊|补偿|赔|各出|谁出|平摊|折中|让步|投票|表决')
NUM_PAT = re.compile(r'\d+(?:\.\d+)?')
BG_TYPES = {'人设'}  # persona / role entries are background and do not count as substantive memory

# D1/D2 counts substantive memory units, not raw anchors. Two deterministic folds come first:
#   1. one unit per turn - parallel fields in one sentence are facets of one statement;
#   2. near-duplicate restatements across turns are merged.
# A unit therefore means a fact established at a distinct moment.
# D1/D2 按实质记忆单元计数而非锚点条数: 先同轮次折叠, 再跨轮次合并近义复述。
D2_MIN_UNITS = int(os.environ.get('D2_MIN_UNITS', '2'))  # this many units or more -> D2
AGG_MIN_UNITS = int(os.environ.get('AGG_MIN_UNITS', '2'))  # threshold for op_type=aggregate
UNIT_SIM = float(os.environ.get('UNIT_SIM', '0.6'))  # similarity threshold for folding across turns


def _norm_fact(s):
    return re.sub(r'[《》「」“”"\'（）()、，。\s]', '', s or '')


def substantive_units(anchors):
    """Substantive memory units (code, no model): drop persona entries -> fold within a turn
    -> fold near-duplicates across turns. Returns the folded anchors; the length is
    n_substantive.
    实质记忆单元: 去人设 → 同轮次折叠 → 跨轮次近重复折叠。
    """
    kept, seen = [], set()
    for a in sorted((x for x in anchors if x.get('type') not in BG_TYPES),
                    key=lambda x: x['turn']):
        if a['turn'] in seen:
            continue
        seen.add(a['turn'])
        kept.append(a)
    out = []
    for a in kept:
        f = _norm_fact(a.get('fact'))
        if any(difflib.SequenceMatcher(None, f, _norm_fact(b.get('fact'))).ratio() > UNIT_SIM
               for b in out):
            continue
        out.append(a)
    return out


SLOT_MIN_SHARED = int(os.environ.get('SLOT_MIN_SHARED', '2'))  # same-slot test: minimum number of shared content bigrams after removing digits
SLOT_STOP = {'我们', '们把', '把这', '这个', '一下', '现在', '大家', '可以', '需要', '请你', '帮我'}
# Unit words alone do not show that two statements are about the same slot ("80k budget"
# and "60k staff" share only the unit), so the shared fingerprint must contain at least
# one non-unit word.
# 只共享计量单位不足以说明是同一槽位, 共享指纹中至少要有一个非计量词。
UNIT_BIGRAMS = {'公里', '千米', '米数', '分钟', '小时', '秒钟', '万元', '块钱', '公斤',
                '毫升', '人次', '天数', '次数'}
# A one-off record is not an updatable slot: an event that already happened does not take
# a new value when a rule changes. Rule: the old anchor names a specific member and
# contains a one-off event word, while the update message does not mention that member.
# Used only to exclude, never to admit.
# 一次性记录(已发生的事件)没有「当前取值」, 不构成可更新的槽位。只用于排除。
EPISODIC_PAT = re.compile(r'此前|已有|已经|提交|发过|发了|这次|本次|上次|一次|当天|那天')


def _members(msgs):
    out = set()
    for m in msgs:
        w = (m.get('who') or '').strip()
        if not w or m.get('kind') == 'bot':
            continue
        out.add(w)
        out.add(re.split(r'[（(]', w)[0])  # "Mr Chen (Chen Lan)" -> "Mr Chen"
    return {x for x in out if len(x) >= 2}


def _is_episodic_record(fact, upd_text, members):
    who = [n for n in members if n in fact]
    return bool(who) and bool(EPISODIC_PAT.search(fact)) and not any(n in upd_text for n in who)


UPD_NEAR = int(os.environ.get('UPD_NEAR', '25'))  # maximum distance (characters) between update wording and the new value
# Only short, focused messages are assigned D3 automatically. A real slot update is a
# sentence about that one thing; in long reports, words like "change to" occur by
# coincidence. Longer messages are not dropped but sent to the arbitration queue.
# 只有短而聚焦的消息才自动判 D3; 长消息里的「改成」多属巧合, 送仲裁队列人工确认。
UPD_MAX_LEN = int(os.environ.get('UPD_MAX_LEN', '80'))


def _upd_near_value(t, new_vals):
    """The update wording must be close to one of the new values; a distant "change to" in a
    long message is no evidence that this slot was updated.
    更新措辞须与某个新值邻近。
    """
    spans = [m.start() for m in UPDATE_PAT.finditer(t)]
    if not spans:
        return False
    for v in new_vals:
        i = t.find(v)
        while i >= 0:
            if any(abs(i - p) <= UPD_NEAR for p in spans):
                return True
            i = t.find(v, i + 1)
    return False


def _slot_bigrams(s):
    """Slot fingerprint: Chinese/English bigrams after removing digits and punctuation, used
    to decide whether two statements refer to the same slot.
    槽位指纹: 去掉数字与标点后的 bigram 集合。
    """
    s = NUM_PAT.sub('', s or '')
    s = re.sub(r'[^\u4e00-\u9fffA-Za-z]', '', s)
    return {s[i:i + 2] for i in range(len(s) - 1)} - SLOT_STOP


def detect_update(anchors, msgs, qturn, members=None):
    """D3-update structural detection (code): an anchor fact holds a changeable value, and a
    later window BEFORE the query contains update wording + a new value + the same slot.
    The upper bound qturn is strict; using information after the query would leak the future.
    Returns a list of (old-value anchor, update turn). Conservative: no detection does not
    mean no conflict, only that D3 cannot be assigned automatically.
    Three ways to show the update refers to the same slot: the old value reappears as a
    whole token, the first 6 characters of the anchor reappear, or at least SLOT_MIN_SHARED
    content bigrams are shared. Old values need at least 2 digits; new values may have 1.
    D3-update 结构检测: 锚点含可变值, 且在请求之前出现「更新措辞 + 新值 + 同一槽位」。检测保守。
    """
    hits, weak = [], []
    members = members if members is not None else _members(msgs)
    for a in anchors:
        # Old values need at least 2 digits: single digits are everywhere in long bot replies.
        # 旧值至少 2 位数字, 个位数在长回复里噪声太多。
        vals = {v for v in NUM_PAT.findall(a.get('fact') or '') if len(v) >= 2}
        if not vals:
            continue
        fact = a.get('fact') or ''
        fact_slot = _slot_bigrams(fact)
        for m in msgs:
            if m['turn'] <= a['turn'] or m['turn'] >= qturn:
                continue
            if m.get('kind') == 'bot':
                continue  # bot-generated content is not group memory and is not a state update
            t = m['text'] or ''
            if not UPDATE_PAT.search(t):
                continue
            new_vals = set(NUM_PAT.findall(t)) - vals  # new values may be a single digit
            if not new_vals:
                continue
            shared = fact_slot & _slot_bigrams(t)
            # All three paths require slot evidence:
            #   1. the old value is matched as a whole number token ("10" does not match "1050");
            #   2. the shared fingerprint must contain a non-unit word (an empty fingerprint never counts);
            #   3. the update wording must be within UPD_NEAR characters of the new value.
            # 三条通路都要求槽位证据: 旧值整 token 匹配; 共享指纹含非计量词; 更新措辞与新值邻近。
            has_slot = bool(shared - UNIT_BIGRAMS)
            t_nums = set(NUM_PAT.findall(t))
            strong = (bool(vals & t_nums)  # old value reappears as a whole token
                      or (fact[:6] and fact[:6] in t)  # first 6 characters of the anchor reappear
                      or len(shared) >= SLOT_MIN_SHARED)  # enough shared slot fingerprint
            same_slot = strong and has_slot and _upd_near_value(t, new_vals)
            if same_slot and not _is_episodic_record(fact, t, members):
                rec = {'old_anchor_turn': a['turn'], 'update_turn': m['turn'],
                       'old_vals': sorted(vals), 'new_vals': sorted(new_vals)[:4],
                       'slot_shared': sorted(shared)[:6], 'upd_len': len(t)}
                if len(t) > UPD_MAX_LEN:
                    weak.append(rec)  # long message: cannot decide, send to arbitration
                else:
                    hits.append(rec)
                break
    detect_update.weak = weak  # picked up by the main loop and sent to arbitration (never dropped silently)
    return hits


def derive_op(n_sub, is_conflict):
    """Derive op_type. n_sub is the number of folded memory units. aggregate only claims that
    several memories must be combined, a weaker claim than D2's, and has its own
    threshold (AGG_MIN_UNITS).
    派生操作类型; aggregate 使用独立门槛 AGG_MIN_UNITS。
    """
    if is_conflict:
        return 'resolve'
    return 'aggregate' if n_sub >= AGG_MIN_UNITS else 'recall'


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    gid = next((a for a in sys.argv[1:] if not a.startswith('-')), 'demo')
    d = rundir(gid)
    msgs = load_msgs(d)
    rows = load_jsonl(os.path.join(d, 'anchors.jsonl'))
    gold = {a['turn']: a for a in load_jsonl(os.path.join(d, 'annotations_gold.jsonl'))}
    if not rows:
        raise SystemExit('anchors.jsonl 为空, 先跑 pipeline/p3_anchor.py')

    # Event clustering (P4-4). Two positions are linked only if they share at least
    # EV_MIN_SHARED anchor turns, their Jaccard is at least EV_MIN_JACCARD, and they are no
    # more than EV_MAX_GAP turns apart. Without these limits, one widely cited background
    # anchor or a chain of neighbouring positions would merge a whole group into one event,
    # and the hub marks and L1<->L2 pairing in P5 that depend on event_id would lose meaning.
    # 事件聚类: 共享锚点数、Jaccard 与时间距离三项同时满足才连通, 避免整群被串成一个事件。
    EV_MIN_SHARED = int(os.environ.get('EV_MIN_SHARED', '2'))
    EV_MIN_JACCARD = float(os.environ.get('EV_MIN_JACCARD', '0.3'))
    EV_MAX_GAP = int(os.environ.get('EV_MAX_GAP', '60'))
    uf = UF()
    aset = {r['turn']: {a['turn'] for a in r['anchors']} for r in rows}
    for t in aset:
        uf.find(t)  # every position starts as its own cluster
    keys = sorted(aset)
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            if y - x > EV_MAX_GAP:
                break  # keys are sorted; everything after is farther away
            inter = aset[x] & aset[y]
            if len(inter) < EV_MIN_SHARED:
                continue
            union_n = len(aset[x] | aset[y])
            if union_n and len(inter) / union_n >= EV_MIN_JACCARD:
                uf.union(x, y)
    eid_map, out_p = {}, os.path.join(d, 'classify.jsonl')
    # Carry over layer values already filled by an earlier run (p4_layer), so recomputing
    # the other fields does not reset them to pending. --reset-layer clears them explicitly.
    # 重算前先保留已填的 layer, 避免被重置为 pending; --reset-layer 可显式清空。
    reset_layer = '--reset-layer' in sys.argv[1:]
    prev_layer = ({} if reset_layer else
                  {r['turn']: r['layer'] for r in load_jsonl(out_p)
                   if isinstance(r.get('layer'), dict) and r['layer'].get('v')})
    if os.path.exists(out_p):
        os.remove(out_p)  # all other fields are recomputed (pure derivation, idempotent)

    n_d = defaultdict(int)
    agree = defaultdict(int)  # agreement counts between rule-derived and human labels (when gold exists)
    for r in rows:
        anchors = r['anchors']
        units = substantive_units(anchors)  # folded substantive memory units
        n_sub = len(units)
        n_raw = sum(1 for a in anchors if a.get('type') not in BG_TYPES)
        upd = detect_update(anchors, msgs, r['turn'])
        for w in getattr(detect_update, 'weak', ()):  # long update messages cannot be decided: send to a human
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P4-D3weak', 'turn': r['turn'],
                          'old_anchor_turn': w['old_anchor_turn'],
                          'update_turn': w['update_turn'], 'upd_len': w['upd_len'],
                          'reason': f"d3_update_in_long_message({w['upd_len']}字, 超 {UPD_MAX_LEN} 上限, 需人判)"})
        arbit_cue = bool(ARBIT_PAT.search((r.get('query') or '') +
                                          ''.join(a.get('fact') or '' for a in anchors)))
        # A D3 position must leave an answerable memory. P5 moves superseded old-value anchors
        # into distractor_atoms; if every anchor of a position is an old value, gt_atoms would be
        # empty. Such a position only remembers values that are no longer valid, so it is not
        # D3 and is graded by unit count instead.
        # D3 位置必须留下可作答的记忆; 若锚点全是被覆盖的旧值, 则不判 D3, 按条数定档。
        old_turns = {h['old_anchor_turn'] for h in upd}
        if upd and old_turns >= {a['turn'] for a in units}:
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P4-D3allstale', 'turn': r['turn'],
                          'old_turns': sorted(old_turns),
                          'reason': 'D3 降级: 全部实质锚点都是旧值锚, 判 D3 会清空 GT'})
            upd = []
        if upd:
            diff = {'v': 'D3', 'by': 'rule:update_detect', 'conflict_kind': 'update',
                    'evidence': upd[:2]}
        else:
            diff = {'v': 'D2' if n_sub >= D2_MIN_UNITS else 'D1', 'by': 'rule:memory_units',
                    'n_units': n_sub, 'n_anchors_raw': n_raw, 'threshold': D2_MIN_UNITS}
        if arbit_cue and not upd:
            # Multi-party disputes need a human: D3 is not assigned automatically; the position is
            # sent to the arbitration queue and flagged here.
            # 多方裁决只能人工: 不自动判 D3, 送仲裁队列并在此留标记。
            diff['arbitration_flag'] = True
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P4-3', 'turn': r['turn'], 'q': r['query'][:60],
                          'reason': 'arbitration_cue(零和裁决需人判)'})
        g = gold.get(r['turn'])
        # Priority: human gold label > value filled in an earlier run > pending
        if g and g['labels'].get('layer', {}).get('v'):
            layer = g['labels']['layer']
        elif r['turn'] in prev_layer:
            layer = prev_layer[r['turn']]
        else:
            layer = {'v': None, 'by': 'pending'}
        root = uf.find(r['turn'])
        eid_map.setdefault(root, f'{gid}-ev{len(eid_map):03d}')
        row = {'turn': r['turn'], 'n_substantive': n_sub, 'n_anchors_raw': n_raw,
               'difficulty': diff,
               'op_type': {'v': derive_op(n_sub, bool(upd)), 'by': 'rule:op_derive'},
               'layer': layer, 'event_id': eid_map[root]}
        n_d[diff['v']] += 1
        gd = g['labels']['difficulty'].get('v') if g and g['labels'].get('difficulty') else None
        if gd:
            agree[f"rule:{diff['v']}/gold:{gd}"] += 1
        append_jsonl(out_p, row)
    ev_sizes = defaultdict(int)
    for r in load_jsonl(out_p):
        ev_sizes[r['event_id']] += 1
    hubs = sum(1 for v in ev_sizes.values() if v >= 2)
    n_kept = len({r['turn'] for r in load_jsonl(out_p)  # deduplicate by turn: older artifacts may contain repeated rows
                  if r['turn'] in prev_layer and (r.get('layer') or {}).get('v')})
    print(f'P4 汇总: 位置 {len(rows)} · D 分布 {dict(n_d)} · 事件 {len(ev_sizes)}(多题事件 {hubs})'
          + (f' · 沿用已填 layer {n_kept} 条' if n_kept else '')
          + (' · --reset-layer 已清空既有 layer' if reset_layer else ''))
    if agree:
        total = sum(agree.values())
        diag = sum(v for k, v in agree.items() if k.split('/')[0][5:] == k.split('/')[1][5:])
        print(f'P4 规则 vs 人工(参考, 非盲标): 一致 {diag}/{total}; 明细 {dict(sorted(agree.items()))}')
    print(f'→ {out_p}')


if __name__ == '__main__':
    main()
