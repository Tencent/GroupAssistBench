#!/usr/bin/env python3
# Canonical acceptance audit: the release red lines (R1-R21 plus one corpus-level check)
# over a finished run directory. Read-only; exits non-zero when a red line fails. Each
# rule's logic and threshold is stated in this file.
#   - R4 fails only when, within one difficulty level, layer, query_origin and op_type
#     each have a single value; the stricter reading (at least two attributes with two
#     or more values) is reported but not enforced.
#   - R7 mechanical pre-screen: Jaccard of the \w-character sets (lower-cased) of
#     old_value / new_value < 0.2. It only queues pairs for a human (status MANUAL) and
#     never counts as a failure.
#   - R15 / R19 also have document-side requirements that a script cannot check; the
#     measured values are printed for comparison.
# Both the full set and the built set are reported; single-run and full-scan modes.
# Usage:
#   python3 tools/audit_canonical.py <run>...     # given run directories (full names, no prefix guessing)
#   python3 tools/audit_canonical.py --full       # every run directory under pipeline_runs/
#   --all      also report observation items (e.g. corpus-level gate output)
#   --verbose  print passing checks as well
# 验收器: 对跑完的 run 目录检查发布红线, 只读; 有红线未过时以非零码退出。每条规则的判定逻辑与阈值都写在本文件中。
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'pipeline_runs')

# ---------- basics ----------

def load_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def norm(s):
    return re.sub(r'[\W_]+', '', str(s or '').lower())


def charset(s):
    return {c for c in str(s or '').lower() if re.match(r'\w', c)}


def jaccard(a, b):
    A, B = charset(a), charset(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


class Ctx:
    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(RUNS, name)
        self.items = load_jsonl(os.path.join(self.dir, 'items.jsonl'))
        self.built = [i for i in self.items if i.get('status') == 'built']
        self.built_ids = {i['iid'] for i in self.built}
        self.gates = load_jsonl(os.path.join(self.dir, 'gates.jsonl'))
        self.arb = load_jsonl(os.path.join(self.dir, 'arbitration_queue.jsonl'))
        self.classify = load_jsonl(os.path.join(self.dir, 'classify.jsonl'))
        self.d3u = load_jsonl(os.path.join(self.dir, 'd3_updates.jsonl'))
        self.msgs = {m['turn']: m for m in load_jsonl(os.path.join(self.dir, 'messages.jsonl'))}
        rm_p = os.path.join(self.dir, 'run_meta.json')
        self.run_meta = json.load(open(rm_p)) if os.path.exists(rm_p) else {}

    def gate_rows(self, rules=None, severity=None, built_only=True):
        out = []
        for r in self.gates:
            if rules is not None and r.get('rule') not in rules:
                continue
            if severity is not None and r.get('severity') != severity:
                continue
            if built_only and r.get('iid') not in self.built_ids:
                continue
            out.append(r)
        return out


# Each check returns (status, detail); status is PASS / FAIL / MANUAL / REPORT / SKIP.

def r1(c):
    """R1 all six cells of the design matrix are non-empty"""
    tab = Counter((i['layer'], i['difficulty']) for i in c.built)
    empty = [f'{l}×{d}' for l in ('L1', 'L2') for d in ('D1', 'D2', 'D3') if not tab[(l, d)]]
    cells = ' '.join(f'{l}/{d}={tab[(l, d)]}' for l in ('L1', 'L2') for d in ('D1', 'D2', 'D3'))
    return ('FAIL' if empty else 'PASS'), (f'空格 {empty} · ' if empty else '') + cells


def r2(c):
    """R2 grading threshold equals the published definition (D2_MIN_UNITS == 2)"""
    thr = Counter(r.get('difficulty', {}).get('threshold') for r in c.classify)
    # threshold exists only for rule:memory_units (D3 uses rule:update_detect and has none)
    bad_thr = {t for t in thr if t is not None and t != 2}
    # Reference count: built D1 items whose anchors span more than one turn (same-turn folding
    # can be recomputed in code, semantic merging cannot; reported only)
    multi = sum(1 for i in c.built if i['difficulty'] == 'D1'
                and len({a['turn'] for a in i.get('anchors', [])}) > 1)
    det = f'classify 阈值分布 {dict(thr)} · D1 built 中多 turn 锚点 {multi} 题(参考, 含语义合并盲区)'
    if not c.classify:
        return 'SKIP', 'classify.jsonl 缺失'
    return ('FAIL' if bad_thr else 'PASS'), det


def r3(c):
    """R3 the number of D3 items follows the material (D3 items / paired conflicts >= 0.6;
    dropped pairs need a reason)
    """
    paired = [u for u in c.d3u if 'old_anchor_turn' in u]
    d3_built = sum(1 for i in c.built if i['difficulty'] == 'D3')
    unpaired = [u for u in c.d3u if 'old_anchor_turn' not in u]
    if not paired:
        return ('PASS' if d3_built == 0 else 'REPORT'), \
            f'无成对冲突对 · D3 built {d3_built}(全部来自配对外供给)'
    no_reason = sum(1 for u in unpaired if not u.get('reason'))
    ratio = d3_built / len(paired)
    ok = ratio >= 0.6 and no_reason == 0
    return ('PASS' if ok else 'FAIL'), \
        (f'D3 built {d3_built} 题 / 成对冲突对 {len(paired)} 组 = {ratio:.2f}(线 ≥0.60) · '
         f'未成对 {len(unpaired)} 组(其中缺 reason {no_reason})')


def r4(c):
    """R4 no difficulty level may be degenerate in all three attributes at once (the
    stricter reading is reported only)
    """
    attrs = ('layer', 'query_origin', 'op_type')
    quick_bad, detail_bad, per = [], [], []
    for d in ('D1', 'D2', 'D3'):
        pool = [i for i in c.built if i['difficulty'] == d]
        if not pool:
            continue
        nv = {a: len({i.get(a) for i in pool}) for a in attrs}
        per.append(f"{d}: " + '/'.join(f'{a}={nv[a]}' for a in attrs))
        if all(v == 1 for v in nv.values()):
            quick_bad.append(d)
        if sum(1 for v in nv.values() if v >= 2) < 2:
            detail_bad.append(d)
    det = (' | '.join(per) + f' · 速查口径(判定依据, D-6)未过: {quick_bad}'
           + f' · 详述口径未过(仅供参考, 不判定): {detail_bad}')
    return ('FAIL' if quick_bad else 'PASS'), det


def r5(c):
    """R5 the only memory of a D1 item must come from a group member"""
    bad = [i['iid'] for i in c.built if i['difficulty'] == 'D1' and i.get('anchors')
           and all(a.get('bot_sourced') for a in i['anchors'])]
    return ('FAIL' if bad else 'PASS'), f'D1 唯一记忆全为 bot 内容 {len(bad)} 题' + \
        (f' {bad[:6]}' if bad else '')


def r6(c):
    """R6 anchors are literally identifiable"""
    bad = c.gate_rows(rules={'anchor_not_identifiable'})
    warn = c.gate_rows(rules={'anchor_unverifiable_no_span'})
    st = 'FAIL' if bad else 'PASS'
    return st, f'anchor_not_identifiable(built)={len(bad)} · 无span warn {len(warn)} 条(补span或降级)'


def r7(c):
    """R7 old and new value of a D3 pair refer to the same slot (mechanical pre-screen only
    queues; a human decides)
    """
    susp = [u for u in c.d3u if jaccard(u.get('old_value'), u.get('new_value')) < 0.2]
    n_susp = len(susp)
    keys = {(u['old_anchor_turn'], u['update_turn']) for u in susp if 'old_anchor_turn' in u}
    on_built = set()
    for i in c.built:
        if i['difficulty'] != 'D3':
            continue
        turns = {a['turn'] for a in i.get('anchors', [])}
        for k in keys:
            if set(k) <= turns:
                on_built.add(i['iid'])
    arb_d3 = sum(1 for r in c.arb if 'd3' in str(r.get('stage', '')).lower())
    if not c.d3u:
        return 'PASS', '无 d3_updates'
    return ('MANUAL' if (n_susp or arb_d3) else 'PASS'), \
        (f'粗筛疑似 {n_susp}/{len(c.d3u)} 组(逐题人工核, 判定权在人) · '
         f'落在 built 题 {len(on_built)} 道 · 仲裁队列 D3 相关 {arb_d3} 条')


def r8(c):
    """R8 bot_sourced matches the transcript"""
    bad = []
    for i in c.built:
        for a in i.get('anchors', []):
            actual = (c.msgs.get(a['turn']) or {}).get('kind') == 'bot'
            if bool(a.get('bot_sourced')) != actual:
                bad.append((i['iid'], a['turn']))
    return ('FAIL' if bad else 'PASS'), f'不一致 {len(bad)} 处' + (f' {bad[:5]}' if bad else '')


def r9(c):
    """R9 no backward causality (and cut_turn <= 2 is not admitted)"""
    causal = [i['iid'] for i in c.built if i.get('anchors')
              and max(a['turn'] for a in i['anchors']) >= (i.get('cut_turn') or 0)]
    early = [i['iid'] for i in c.built if (i.get('cut_turn') or 99) <= 2]
    g = c.gate_rows(rules={'future_ref', 'premise_after_cut', 'premise_span_not_found'},
                    severity='error')
    bad = causal or early or g
    return ('FAIL' if bad else 'PASS'), \
        f'锚点≥cut {len(causal)} · cut_turn≤2 {len(early)} · 闸门(error,built) {len(g)}'


R10_ERR = {'d3_query_self_contained', 'd3_query_quotes_both_values', 'self_answer_quote',
           'query_self_answer'}

def r10(c):
    """R10 the query does not contain the answer: decided by the gates; containment matches
    go to a human
    """
    err = c.gate_rows(rules=R10_ERR)
    # Recomputed pre-screen: the query's net text and an anchor's fact / evidence_span
    # contain one another (>= 6 characters)
    mutual = []
    for i in c.built:
        q = norm(i.get('query'))
        for a in i.get('anchors', []):
            for s in (a.get('fact'), a.get('evidence_span')):
                t = norm(s)
                if len(t) >= 6 and (t in q or (len(q) >= 6 and q in t)):
                    mutual.append(i['iid'])
                    break
            else:
                continue
            break
    warn = c.gate_rows(rules={'query_self_answer_contains', 'query_contains_gt_value'})
    st = 'FAIL' if err else ('MANUAL' if (mutual or warn) else 'PASS')
    return st, (f'闸门层(built) {len(err)} · 互含粗筛复算 {len(mutual)} 题(送人工) · '
                f'互含/含GT warn {len(warn)} 条(送人工)')


def r11(c):
    """R11 no dimension-name or hindsight leakage"""
    err = c.gate_rows(rules={'dimension_name_leak', 'hindsight_wording'})  # rule names must match those emitted by pipeline/gates.py
    warn = c.gate_rows(rules={'dimension_named', 'posthoc_framing'})
    st = 'FAIL' if err else ('MANUAL' if warn else 'PASS')
    return st, f'error 级 {len(err)} · warn 逐条人工 {len(warn)} 条'


def r12(c):
    """R12 answer fields are present"""
    bad = [i['iid'] for i in c.built
           if (i['layer'] == 'L1' and not (i.get('gt') and i.get('gt_atoms')))
           or (i['layer'] == 'L2' and not i.get('criteria'))]
    return ('FAIL' if bad else 'PASS'), f'缺答案字段 {len(bad)} 题' + (f' {bad[:6]}' if bad else '')


def r13(c):
    """R13 L2 criteria cover all three kinds"""
    bad = []
    for i in c.built:
        if i['layer'] != 'L2':
            continue
        kinds = {cr.get('kind') for cr in i.get('criteria') or []}
        if not {'grounding', 'generic', 'penalty'} <= kinds:
            bad.append((i['iid'], sorted({'grounding', 'generic', 'penalty'} - kinds)))
    return ('FAIL' if bad else 'PASS'), f'三类不齐 {len(bad)} 题' + (f' {bad[:4]}' if bad else '')


def r14(c):
    """R14 criteria count and polarity contract"""
    bad = []
    for i in c.built:
        if i['layer'] != 'L2':
            continue
        cr = i.get('criteria') or []
        pos = [x for x in cr if (x.get('points') or 0) > 0]
        pen = [x for x in cr if x.get('kind') == 'penalty']
        probs = []
        if not 3 <= len(pos) <= 6:
            probs.append(f'正分{len(pos)}条')
        if len(pen) > 3:
            probs.append(f'penalty{len(pen)}条')
        if len(cr) > 8:
            probs.append(f'总数{len(cr)}')
        if any((x.get('points') or 0) >= 0 for x in pen):
            probs.append('penalty非负分')
        if any(not 1 <= (x.get('points') or 0) <= 3 for x in pos):
            probs.append('正分越界')
        if any((x.get('coverage_n') or 0) > 2 for x in cr if x.get('kind') == 'grounding'):
            probs.append('coverage_n>2')
        if probs:
            bad.append((i['iid'], probs))
    return ('FAIL' if bad else 'PASS'), f'违约 {len(bad)} 题' + (f' {bad[:4]}' if bad else '')


def r15(c):
    """R15 effective number of independent samples (three conversions, reported next to the
    item count)
    """
    n = len(c.built)
    if not n:
        return 'SKIP', '无 built 题'
    by_anchor = defaultdict(list)
    for i in c.built:
        key = frozenset((a['turn'], str(a.get('fact', ''))[:14]) for a in i.get('anchors', []))
        by_anchor[key].append(i['iid'])
    by_tpl = defaultdict(list)
    for i in c.built:
        by_tpl[norm(i.get('query'))[:60]].append(i['iid'])
    ev = Counter(i.get('event_id') for i in c.built if i.get('event_id'))
    eff = min(len(by_anchor), len(by_tpl), len(ev) or n)
    dup_a = sum(len(v) for v in by_anchor.values() if len(v) > 1)
    dup_t = sum(len(v) for v in by_tpl.values() if len(v) > 1)
    return 'REPORT', (f'题数 {n} · 有效独立样本 ≤{eff} '
                      f'(同锚点集折算 {len(by_anchor)}, 同题面前60字折算 {len(by_tpl)}, '
                      f'唯一 event {len(ev)}, 最大簇 {max(ev.values()) if ev else 0}) · '
                      f'涉重复题: 同锚 {dup_a} / 同模板 {dup_t}')


def r16(c):
    """R16 every number in run_meta equals the actual count"""
    p = c.run_meta.get('pipeline') or {}
    if not p:
        return 'SKIP', 'run_meta 无 pipeline 段'
    mism = []

    def cmp(name, meta_v, actual_v):
        if meta_v is not None and meta_v != actual_v:
            mism.append(f'{name}: meta={meta_v} 实际={actual_v}')

    cmp('items', p.get('items'), len(c.items))
    cmp('items_by_status', p.get('items_by_status'),
        dict(Counter(i['status'] for i in c.items)))
    cmp('items_by_layer', p.get('items_by_layer'), dict(Counter(i['layer'] for i in c.items)))
    cmp('items_by_difficulty', p.get('items_by_difficulty'),
        dict(Counter(i['difficulty'] for i in c.items)))
    cmp('items_by_track', p.get('items_by_track'), dict(Counter(i.get('track') for i in c.items)))
    cmp('items_by_query_origin', p.get('items_by_query_origin'),
        dict(Counter(i.get('query_origin') for i in c.items)))
    cmp('candidates', p.get('candidates'),
        len(load_jsonl(os.path.join(c.dir, 'candidates.jsonl'))))
    cmp('classified', p.get('classified'), len(c.classify))
    inv_p = os.path.join(c.dir, 'anchor_inventory.json')
    if os.path.exists(inv_p) and p.get('inventory') is not None:
        cmp('inventory', p.get('inventory'), len(json.load(open(inv_p))))
    cmp('gate_findings', p.get('gate_findings'),
        dict(Counter(g.get('severity') for g in c.gates)))
    cmp('gate_errors', p.get('gate_errors'),
        sum(1 for g in c.gates if g.get('severity') == 'error'))
    cmp('arbitration_queue', p.get('arbitration_queue'), len(c.arb))
    cmp('arbitration_by_stage', p.get('arbitration_by_stage'),
        dict(Counter(a.get('stage') for a in c.arb)))
    # refreshed_at must be later than the last data change (a .bak file's mtime is a lower
    # bound for that moment)
    ra = c.run_meta.get('refreshed_at')
    if ra:
        import datetime
        try:
            ts = datetime.datetime.fromisoformat(str(ra).replace(' ', 'T')).timestamp()
            for b in glob.glob(os.path.join(c.dir, '*.bak*')):
                if os.path.getmtime(b) > ts:
                    mism.append(f'refreshed_at({ra}) 早于 {os.path.basename(b)}')
        except ValueError:
            mism.append(f'refreshed_at 不可解析: {ra}')
    return ('FAIL' if mism else 'PASS'), ('; '.join(mism[:6]) or '逐项一致') + \
        (f' …共 {len(mism)} 项' if len(mism) > 6 else '')


def r17(c):
    """R17 no error finding on a built item"""
    hit = sorted({g['iid'] for g in c.gates
                  if g.get('severity') == 'error' and g.get('iid') in c.built_ids})
    return ('FAIL' if hit else 'PASS'), f'error∩built = {len(hit)}' + (f' {hit[:6]}' if hit else '')


def r18(c):
    """R18 the audit trail is append-only: entries of the arbitration queue (the only input
    to human review) may be added but never lost. gates.jsonl mirrors the latest gate run,
    so findings that disappear after a fix are reported, not judged.
    """
    def losses(fname, rows, key):
        cur = Counter(r.get(key) for r in rows)
        out = []
        for b in sorted(glob.glob(os.path.join(c.dir, fname + '.bak*'))):
            old = Counter(r.get(key) for r in load_jsonl(b))
            lost = {k: v - cur.get(k, 0) for k, v in old.items() if cur.get(k, 0) < v}
            if lost:
                out.append(f'{os.path.basename(b)} → 丢 {lost}')
        return out
    arb_loss = losses('arbitration_queue.jsonl', c.arb, 'stage')
    gate_loss = losses('gates.jsonl', c.gates, 'rule')
    if not glob.glob(os.path.join(c.dir, '*.bak*')):
        return 'PASS', '无 .bak 对照(未发生重建)'
    det = ('仲裁队列: ' + ('; '.join(arb_loss[:3]) if arb_loss else '各 stage 只增不减')
           + ' · gates(参考): ' + ('; '.join(gate_loss[:2]) if gate_loss else '无减项'))
    return ('FAIL' if arb_loss else ('MANUAL' if gate_loss else 'PASS')), det


def r19(c):
    """R19 release documents and artifacts come from the same round; needs the documents,
    so a script cannot decide it
    """
    return 'SKIP', ('须对照交付文档逐数核对(本脚本报出实际值: '
                    f"题 {len(c.items)} / built {len(c.built)} / "
                    f"flagged {sum(1 for i in c.items if i.get('status') == 'flagged')} / "
                    f"gate_error {sum(1 for g in c.gates if g.get('severity') == 'error')})")


def r20(c):
    """R20 a single authoritative version of each artifact"""
    extras = [os.path.basename(p) for p in glob.glob(os.path.join(c.dir, 'items_v*.jsonl'))]
    main = os.path.join(c.dir, 'items.jsonl')
    bad = []
    for e in extras:
        if open(os.path.join(c.dir, e)).read() != open(main).read():
            bad.append(e)
    return ('FAIL' if bad else 'PASS'), \
        (f'并存且内容不同: {bad}' if bad else ('目录下仅一份 items.jsonl' if not extras
                                              else f'{extras} 与正式版内容一致'))


def r21(c):
    """R21 structural invariants of revised D3 items (item.d3v2 present): grounding >= 2
    (L2), anchors >= 3, speakers >= 2, no timeline cue words in the query
    (gates.D3_HINT_PAT). SKIP when there are no revised D3 items.
    """
    import sys as _sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from pipeline.gates import D3_HINT_PAT  # single source; the regex is not copied here
    who = {t: m.get('who') for t, m in c.msgs.items()} if getattr(c, 'msgs', None) else {}
    v2 = [i for i in c.built if i.get('d3v2')]
    if not v2:
        return 'SKIP', '无 D3 v2 题'
    bad = []
    for i in v2:
        n_g = sum(1 for x in (i.get('criteria') or []) if x.get('kind') == 'grounding')
        spk = {who.get(a['turn']) for a in i.get('anchors') or []} - {None}
        why = []
        if i['layer'] == 'L2' and n_g < 2:
            why.append(f'grounding={n_g}')
        if len(i.get('anchors') or []) < 3:
            why.append(f"anchors={len(i.get('anchors') or [])}")
        if who and len(spk) < 2:
            why.append(f'speakers={len(spk)}')
        if i['d3v2'].get('form', 'service') == 'service' and D3_HINT_PAT.search(i.get('query') or ''):
            why.append('hint_words')  # trace-form queries may legitimately mention the evolution: exempt
        if why:
            bad.append((i['iid'], why))
    return ('FAIL' if bad else 'PASS'), (f'v2 题 {len(v2)} · 违反 {len(bad)}: {bad[:4]}' if bad
                                        else f'v2 题 {len(v2)} 全部满足 grounding≥2/锚点≥3/发言人≥2/无线索词')


def obs_corpus_gates(c):
    """Corpus-level gates must actually have run (rows with the _corpus_ prefix in gates.jsonl)"""
    n = sum(1 for g in c.gates if str(g.get('iid', '')).startswith('_corpus_'))
    return ('PASS' if n else 'MANUAL'), f'_corpus_* 条目 {n} 条' + ('' if n else ' → 语料级闸门未落盘')


CHECKS = [('R1', r1), ('R2', r2), ('R3', r3), ('R4', r4), ('R5', r5), ('R6', r6),
          ('R7', r7), ('R8', r8), ('R9', r9), ('R10', r10), ('R11', r11), ('R12', r12),
          ('R13', r13), ('R14', r14), ('R15', r15), ('R16', r16), ('R17', r17),
          ('R18', r18), ('R19', r19), ('R20', r20), ('R21', r21)]
MARK = {'PASS': '✓', 'FAIL': '✗', 'MANUAL': '?', 'REPORT': '○', 'SKIP': '-'}


def audit_run(name, verbose=False, extra=False):
    c = Ctx(name)
    checks = CHECKS + ([('B+', obs_corpus_gates)] if extra else [])
    results = [(rid, *fn(c)) for rid, fn in checks]
    n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
    n_man = sum(1 for _, s, _ in results if s == 'MANUAL')
    print('=' * 96)
    print(f"{c.run_meta.get('gid', name)}   {name}   题 {len(c.items)} "
          f"(built {len(c.built)} / flagged {sum(1 for i in c.items if i.get('status') == 'flagged')})"
          f"   红线未过 {n_fail} · 待人工 {n_man}")
    print('=' * 96)
    for rid, st, det in results:
        if st == 'PASS' and not verbose:
            continue
        print(f'  {MARK[st]} {rid:4s}  {det}')
    return n_fail


def main():
    args = [a for a in sys.argv[1:]]
    verbose = '--verbose' in args
    extra = '--all' in args
    names = [a for a in args if not a.startswith('--')]
    if '--full' in args:
        names = sorted(os.path.basename(d) for d in glob.glob(os.path.join(RUNS, '*'))
                       if os.path.isdir(d) and not os.path.basename(d).startswith('_')
                       and '取代' not in os.path.basename(d)
                       and os.path.exists(os.path.join(d, 'items.jsonl')))
    if not names:
        raise SystemExit('用法: audit_canonical.py <run目录名>... | --full  [--all] [--verbose]')
    total_fail = 0
    for n in names:
        total_fail += audit_run(n, verbose=verbose, extra=extra)
    print(f'\n合计: {len(names)} 个 run · 红线未过 {total_fail}')
    sys.exit(1 if total_fail else 0)


if __name__ == '__main__':
    main()
