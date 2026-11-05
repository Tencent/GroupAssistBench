#!/usr/bin/env python3
# Deterministic quality gates (paper 3.2 "quality control"), zero API: structural lint
# plus leakage scans - anchors must precede the cutoff, spans must be found verbatim,
# and a query must not reveal a required memory or hint at a revision.
# Each finding: {iid, gate, rule, severity (error|warn|info), detail}
#   error = violation, must be 0 in the released set; warn = needs a human look;
#   info = statistics only.
# 确定性质量闸门(不调 API): 结构检查 + 泄漏扫描。error 在发布集中必须为 0; warn 需人工查看; info 仅统计。
import difflib
import os
import re
from collections import defaultdict

REF_PAT = re.compile(r'#(\d+)')
META_PAT = re.compile(r'必须明确提及|必须指出|明确提及|必须复述')
HINDSIGHT_PAT = re.compile(r'重蹈|当时答错|事后(?:来看|才知道)')  # words that are literal content in some groups are deliberately not listed
TRACE_TPL = re.compile(r'怎么定|定下来的|是怎么|怎么来的|怎么确定')
QUOTE = re.compile(r'[「『“](.+?)[」』”]')
NUM = re.compile(r'\d+(?:\.\d+)?')
# Anonymised ids (user_72e841, d14294, #123) are removed before value matching, so their
# digits are not mistaken for answer values. To avoid removing real values, a token must
# contain both letters and digits, be at least 4 characters long, and stand alone (no
# adjacent ASCII letters or digits). Plain numbers such as dates, short model names such
# as "T1", and phrases such as "Redmi Note 10" are kept.
# 数值比对前先去掉匿名 ID; 要求同时含字母与数字、长度 ≥4 且是独立 token, 避免误删真实数值。
USER_ID = re.compile(r'user_?[0-9a-z]{2,}|#\d+'
                     r'|(?<![0-9a-z])(?=[0-9a-z]*[a-z])(?=[0-9a-z]*\d)[0-9a-z]{4,}(?![0-9a-z])',
                     re.I)

# ---- Scope guidance may give a count and a form, but not the dimension names ----
# 范围引导只能给数量与形式, 不能给出维度名
DIM_PATTERNS = (
    re.compile(r'从(.{2,20}?)(?:两|三|几)?(?:方面|类|个角度)'),  # "from the two aspects of emotion and behaviour" / "from ... two kinds"
    re.compile(r'按[「“"『]?(.{2,30}?)[」”"』]?(?:两|三|四)点'),  # "by the three points 'identity, task, need'"
    re.compile(r'分(?:别|[两三四五]?点)?说明(.{2,40})'),  # "explain A, B and C respectively" / "explain in three points ..."
    # "Around X," is not matched: X is the topic of the question, and overlap between the
    # topic and the answer is needed to locate the memory; it is not a leak.
    # 「围绕 X」不在此列: X 是题目话题, 与答案重合是定位所需, 不算泄漏。
)
DIM_SPLIT = re.compile(r'、|和|与|及|以及|[,，/]')
DIM_STOP = {'两', '三', '四', '这', '那', '什么', '如何', '怎么', '要求', '情况', '问题',
            '关系', '方面', '内容', '原因', '先后', '顺序'}

def _dim_terms(query):
    terms = []
    for pat in DIM_PATTERNS:
        for m in pat.finditer(query):
            for t in DIM_SPLIT.split(m.group(1)):
                t = t.strip('的 “”"「」『』？?。 ')
                if 2 <= len(t) <= 10 and t not in DIM_STOP:
                    terms.append(t)
    return terms


def _crit(it):
    return [(c.get('kind'), str(c.get('check', ''))) for c in (it.get('criteria') or [])]


def q1_item(it):
    """Q1 structural lint: fields, counts, wording.
    Q1 结构检查: 字段、条数、措辞。
    """
    F = []
    add = lambda rule, sev, detail='': F.append(
        {'iid': it['iid'], 'gate': 'Q1', 'rule': rule, 'severity': sev, 'detail': detail})
    crits = it.get('criteria') or []
    if it['layer'] == 'L2' and crits:
        # Count limits come from the annotation protocol. Auto-generated rubrics are reported as
        # warn rather than info so that violations stay visible.
        # 条数上限来自标注规范; 自动生成的评分标准记 warn 而非 info, 让违规保持可见。
        sev = 'error' if it['criteria_origin'] == 'annotation' else 'warn'
        pos = [c for c in crits if int(c['points']) > 0]
        pen = [c for c in crits if int(c['points']) < 0]
        if not (3 <= len(pos) <= 6):
            add('count_pos_out_of_range', sev, f'pos={len(pos)}')
        if len(pen) > 3:
            add('count_penalty_gt3', sev, f'pen={len(pen)}')
        if len(crits) > 8:
            add('count_total_gt8', sev, f'total={len(crits)}')
    if it['layer'] == 'L2' and crits and not [c for c in crits if int(c['points']) < 0]:
        # An L2 item without any penalty criterion is scored on positive coverage alone, which
        # disables half of the rubric; it is not passed silently.
        # L2 题没有任何扣分项时, 评分只剩正向覆盖, 不能静默放行。
        add('no_penalty_criteria', 'warn', 'L2 无扣分项(疑 profile 退化, 见 2-7)')
    if it.get('layer_provisional'):
        # P5 builds items as L1 when the level could not be decided; such items must not pass silently.
        # P5 判不出层级时按 L1 构题, 这类题不能静默入库。
        add('layer_provisional', 'warn', f"note={it.get('layer_note')}")
    for kind, txt in _crit(it):
        if META_PAT.search(txt):
            add('meta_description', 'error', txt[:42])
    # Two-level check of dimension names
    terms = _dim_terms(it['query'])
    if terms:
        gt_blob = ''.join(it.get('gt_atoms') or []) + (it.get('gt') or '')
        leak = [t for t in terms if t in gt_blob]
        if leak:
            add('dimension_name_leak', 'error', '/'.join(leak[:4]))
        else:
            add('dimension_named', 'warn', '/'.join(terms[:4]))
    return F


def q2_item(it):
    """Q2 leakage scan: (a) citations beyond the cutoff, (b) a query that answers itself,
    (c) hindsight wording.
    Q2 泄漏扫描: 引用越过截断点、题面自答、事后视角措辞。
    """
    F = []
    add = lambda rule, sev, detail='': F.append(
        {'iid': it['iid'], 'gate': 'Q2', 'rule': rule, 'severity': sev, 'detail': detail})
    cut = it.get('cut_turn')
    texts = [t for _, t in _crit(it)] + list(it.get('gt_atoms') or []) \
        + [it.get('gt') or ''] + [str(x) for x in (it.get('gt_refs') or [])]
    if cut:
        refs = sorted({int(m) for t in texts for m in REF_PAT.findall(t)})
        bad = [r for r in refs if r > int(cut)]
        if bad:
            add('future_ref', 'error', f'cut={cut} refs={bad[:5]}')
        # Legacy items store anchors as a list of ints; P5 items as a list of {turn, fact, ...}
        # dicts. Both shapes are checked.
        # 旧题的锚点是整数列表, P5 新题是字典列表, 两种都检查。
        a_turns = [a if isinstance(a, int) else a.get('turn')
                   for a in (it.get('anchors') or [])]
        bad_a = [t for t in a_turns if isinstance(t, int) and t > int(cut)]
        if bad_a:
            add('anchor_after_cut', 'error', f'cut={cut} anchors={bad_a[:5]}')
    if it['layer'] == 'L2':
        for kind, txt in _crit(it):
            if kind != 'grounding':
                continue
            # Quoting the query to explain a criterion is not self-answering; the `exposed`
            # treatment is a deliberate A/B control and is exempt.
            if '问的是' in txt or 'query' in txt.lower() or it['query_treatment'] == 'exposed':
                continue
            for span in QUOTE.findall(txt):
                s = span.strip()
                if len(s) >= 4 and s in it['query']:
                    add('self_answer_quote', 'error', s[:30])
    else:
        # An L1 query containing an answer value is the most common form of self-answering;
        # identifier digits are removed first.
        # L1 题面含答案数值是最常见的自答形态; 先剔除标识符中的数字。
        q = USER_ID.sub('', it['query'])
        gt = USER_ID.sub('', ''.join(it.get('gt_atoms') or []))
        hits = sorted({n for n in NUM.findall(gt) if len(n) >= 2 and n in q})
        if hits:
            add('query_contains_gt_value', 'warn', '/'.join(hits[:5]))
    for kind, txt in _crit(it):
        if HINDSIGHT_PAT.search(txt):
            add('hindsight_wording', 'error', txt[:42])
    return F


D3_QUERY_MAX = int(os.environ.get('D3_QUERY_MAX', '75'))
# L2-form D3 items get their own, longer limit: they describe a service request and its
# deliverable, which is structurally longer than an L1 question asking which value holds.
# Leakage itself is caught by d3_query_quotes_both_values, which does not depend on length.
# L2 形态的 D3 题单设更长的长度上限; 泄漏另由与长度无关的「引述两段原话」规则检查。
D3_QUERY_MAX_L2 = int(os.environ.get('D3_QUERY_MAX_L2', '120'))
QUOTED = re.compile(r'[「“"\u2018\']([^」”"\u2019\']{4,})[」”"\u2019\']')


# Revised D3 queries must not contain timeline cue words ("changed", "updated", "now",
# "final", ...): such words tell the model to look for the latest value. The query should
# read like a request from someone who only knows the current arrangement. Applies only
# to revised items (item.d3v2 present).
# 新版 D3 题面不得含时间线索词, 否则等于提示模型去找最新值; 只对新版 D3 题生效。
D3_HINT_PAT = re.compile(r'改过|改了|更改|更新|换成|换了|替换|最新|此刻|现在生效|最终|后来|刚才|之前|原来|原先|上一版|旧的|新的|变更|调整后|定下来的|最后定')


def q2_d3_leak(it):
    """Q2(d) self-answering check for update-type conflict items.
    When an authored query quotes both conflicting values, it can be answered with the
    generic rule "the later statement wins", without any memory. Leakage of this kind is
    paraphrase rather than copying, so similarity to the answer does not detect it; query
    length and the number of quoted segments do.
    Applies only to authored items with conflict_kind=update; native queries are the
    group's own words.
    冲突更新题的题面自答检查: 题面同时引述了新旧两个值时, 不靠记忆也能答对。
    用题面长度与引述段数判断; 只对构造题生效。
    """
    if it.get('conflict_kind') != 'update' or it.get('query_origin') != 'authored':
        return []
    q = it.get('query') or ''
    quoted = [x for x in QUOTED.findall(q) if len(x) >= 4]
    F = []
    cap = D3_QUERY_MAX_L2 if it.get('layer') == 'L2' else D3_QUERY_MAX  # L2-form D3 items use the longer limit
    if len(q) > cap:
        F.append({'iid': it['iid'], 'gate': 'Q2', 'rule': 'd3_query_self_contained',
                  'severity': 'error',
                  'detail': f'题面 {len(q)} 字 > {cap}({it.get("layer")} 档), '
                            f'冲突两端可能都被引述进题面'})
    if len(quoted) >= 2:
        F.append({'iid': it['iid'], 'gate': 'Q2', 'rule': 'd3_query_quotes_both_values',
                  'severity': 'error',
                  'detail': f'题面引述了 {len(quoted)} 段原话: ' + ' / '.join(x[:16] for x in quoted[:2])})
    if it.get('d3v2') and it['d3v2'].get('form', 'service') == 'service':  # revised D3 service form only; the trace form is exempt
        hits = D3_HINT_PAT.findall(q)
        if hits:
            F.append({'iid': it['iid'], 'gate': 'Q2', 'rule': 'd3_query_hint_words',
                      'severity': 'error', 'detail': 'v2 题面含时间/版本线索词: ' + '/'.join(hits[:4])})
    return F


# ---- Self-answering screen: query vs. anchor text ----
# Two signals, both warn + human review, never an automatic flag:
#   1. containment (high precision): one side's net text is a contiguous substring of
#      the other and at least MIN characters long - the typical self-answering form.
#   2. similarity (high recall): normalised SequenceMatcher >= SIM. Template-heavy
#      groups produce false positives, so this signal may never be an error.
# They are kept separate because their false-positive rates differ by an order of
# magnitude, and reviewers need to queue them by reliability.
# 题面与锚点原文的自答粗筛: 互含(高精度)与相似度(高召回)两层, 都只记 warn 送人工, 分开记录。
QSA_MIN_LEN = int(os.environ.get('QSA_MIN_LEN', '6'))
QSA_SIM = float(os.environ.get('QSA_SIM', '0.76'))
_QSA_STRIP = re.compile(r'[^\u4e00-\u9fffA-Za-z0-9]')


def _net(s):
    """Net text: punctuation, whitespace and anonymised ids removed; ids are processing
    artifacts and must not affect the check.
    净文本: 去掉标点、空白与匿名 ID。
    """
    return _QSA_STRIP.sub('', USER_ID.sub('', s or ''))


def q2_query_self_answer(it):
    F = []
    q = _net(it.get('query'))
    if len(q) < QSA_MIN_LEN:
        return F
    best = None
    for a in it.get('anchors') or []:
        if not isinstance(a, dict):
            continue
        for field in ('fact', 'evidence_span'):
            s = _net(a.get(field))
            if len(s) < QSA_MIN_LEN:
                continue
            if s in q or q in s:
                F.append({'iid': it['iid'], 'gate': 'Q2', 'rule': 'query_self_answer_contains',
                          'severity': 'warn',
                          'detail': f"#{a.get('turn')} {field} 与题面互含: {s[:28]}"})
                return F  # one finding is enough to queue the item
            r = difflib.SequenceMatcher(None, q, s).ratio()
            if best is None or r > best[0]:
                best = (r, a.get('turn'), field, s)
    if best and best[0] >= QSA_SIM:
        r, t, field, s = best
        F.append({'iid': it['iid'], 'gate': 'Q2', 'rule': 'query_self_answer_sim',
                  'severity': 'warn',
                  'detail': f'#{t} {field} 与题面相似度 {r:.2f}≥{QSA_SIM}: {s[:28]}'})
    return F


# ---- An item's only memory must not come from the bot ----
# A bot restating an objective group fact may serve as memory. But when ALL of an item's
# anchors are bot turns, the item tests what the assistant said, not group memory, so this
# is an error and the item is excluded from the release. Covers a single bot anchor and
# several anchors that are all bot turns; D3 items are included.
# 题目的全部锚点都来自 bot 时, 考的是助手自己说过什么而非群记忆, 判 error 并排除。
def q7_bot_only_memory(it, bot_turns=frozenset()):
    anchors = [a for a in (it.get('anchors') or []) if isinstance(a, dict)]
    if not anchors:
        return []
    is_bot = lambda a: bool(a.get('bot_sourced')) or a.get('turn') in bot_turns
    if not all(is_bot(a) for a in anchors):
        return []
    return [{'iid': it['iid'], 'gate': 'Q7', 'rule': 'memory_all_bot_sourced',
             'severity': 'error',
             'detail': f"{it.get('difficulty')} 全部 {len(anchors)} 条锚点均为 bot 内容: "
                       + '/'.join(f"#{a.get('turn')}" for a in anchors[:4])}]


def q7_item(it, by_turn, prof_by_turn, bot_turns=frozenset()):
    """Q7 anchor authenticity (hard gate for RT2, deterministic): each anchor must be
    literally identifiable in the turn it cites.
      evidence_span found in the message or in a profile entry -> pass
      span present but not found -> error (mismatched anchor)
      span missing -> warn (semantic fallback / human)
    An anchor on a bot turn -> see anchor_on_bot_content below.
    Q7 锚点真实性: 锚点须能在所指轮次中逐字找到; 找不到判 error, 缺片段判 warn。
    """
    F = []
    for a in it.get('anchors') or []:
        if not isinstance(a, dict):
            continue
        t, span, fact = a.get('turn'), a.get('evidence_span') or '', a.get('fact') or ''
        # items.anchors[].bot_sourced must equal kind == 'bot' in the transcript; checked here
        # independently so the gate itself guarantees the field matches the conversation.
        # 锚点的 bot_sourced 必须与原文的 kind 一致, 由闸门独立校验。
        if bool(a.get('bot_sourced')) != (t in bot_turns):
            F.append({'iid': it['iid'], 'gate': 'Q7', 'rule': 'bot_sourced_mismatch',
                      'severity': 'warn',
                      'detail': f"#{t} 标 {bool(a.get('bot_sourced'))}, 语料 kind=bot "
                                f"为 {t in bot_turns}"})
        if t in bot_turns:
            # A bot-turn anchor may be a valid group fact or an invalid knowledge answer; a rule
            # cannot tell them apart, so it is a warn for human confirmation, not an error.
            # bot 行锚点可能有效也可能无效, 规则分不开, 记 warn 送人工确认。
            F.append({'iid': it['iid'], 'gate': 'Q7', 'rule': 'anchor_on_bot_content',
                      'severity': 'warn', 'detail': f'#{t} {fact[:24]}'})
            continue
        txt = by_turn.get(t) or ''
        if span:
            from pipeline.llm import span_in
            ok = span_in(span, txt) or any(span_in(span, p.get('quote')) or span_in(span, p.get('fact'))
                                           for p in prof_by_turn.get(t, []))
            if not ok:
                F.append({'iid': it['iid'], 'gate': 'Q7', 'rule': 'anchor_not_identifiable',
                          'severity': 'error', 'detail': f'#{t} {fact[:24]}'})
        else:
            F.append({'iid': it['iid'], 'gate': 'Q7', 'rule': 'anchor_unverifiable_no_span',
                      'severity': 'warn', 'detail': f'#{t} {fact[:24]}'})
    return F


POSTHOC_PAT = re.compile(r'后来|最后|后面|之后|随后|事后|再后|接着(?:又|还)|然后(?:又|还)')


def q8_posthoc(it):
    """Q8 addition: hindsight wording in authored queries.
    A query can have only valid premises and still narrate events that happen after the
    moment it is asked ("later you also told me ..."); at that moment no answer exists yet.
    Wording is used only as a fallback signal: if every premise carries a turn at or
    before cut_turn, the time boundary is already guaranteed by the premises and "later"
    merely describes the order of earlier events, so nothing is reported. Native queries
    are the group's own words and are not checked.
    构造题题面的事后视角措辞检查。前提全部在截断点之内时不报; 只在缺少可校验前提时以措辞兜底。
    """
    if it.get('query_origin') != 'authored':
        return []
    cut = it.get('cut_turn')
    pts = [p.get('turn') for p in (it.get('premises') or []) if isinstance(p.get('turn'), int)]
    if pts and isinstance(cut, int) and max(pts) <= cut:
        return []  # all premises are within the cutoff: wording is not evidence of a leak
    m = POSTHOC_PAT.search(it.get('query') or '')
    if not m:
        return []
    return [{'iid': it['iid'], 'gate': 'Q8', 'rule': 'posthoc_framing', 'severity': 'error',
             'detail': f'题面含事后视角措辞「{m.group()}」'}]


def q8_item(it, by_turn):
    """Q8 premise authenticity (hard gate for RT2, deterministic): every premise of a query
    carries {turn, span}; the span must be found verbatim in that turn and turn <= cut_turn.
    Q8 前提真实性: 每个前提须带 {turn, span}, 片段在该轮次中逐字可见且不晚于截断点。
    """
    F = []
    add = lambda rule, sev, detail='': F.append(
        {'iid': it['iid'], 'gate': 'Q8', 'rule': rule, 'severity': sev, 'detail': detail})
    prem = it.get('premises')
    if prem is None:
        add('premises_missing', 'warn', '构造题无前提清单')
        return F
    cut = it.get('cut_turn')
    for p in prem:
        t, span = p.get('turn'), p.get('span') or ''
        label = str(p.get('text', ''))[:24]
        if not isinstance(t, int) or not span:
            add('premise_unlocated', 'error', label)
        elif cut and t > int(cut):
            add('premise_after_cut', 'error', f'#{t} {label}')
        elif span not in (by_turn.get(t) or ''):
            add('premise_span_not_found', 'error', f'#{t} {label}')
    return F


def corpus_checks(items):
    """Corpus-level statistics: template share (T8) and atom citation coverage (T2).
    Called by p5_build after self_gate on the full item set. p5 calls it once per pass
    (rt1 / rt2d3 / rt2), so one iid may have several rows; the row with the largest n_items
    (the last full pass) is authoritative. With zero L1 items there is no division by zero.
    跨题统计: 模板占比与原子引用覆盖率; 以 n_items 最大的一行为准。
    """
    F = []
    meta = {'n_items': len(items), 'at': __import__('time').strftime('%F %T')}
    by_d = defaultdict(list)
    for it in items:
        if it['layer'] == 'L1':
            by_d[it['difficulty']].append(it)
    for d, lst in sorted(by_d.items()):
        n = sum(1 for it in lst if TRACE_TPL.search(it['query']))
        share = n / len(lst)
        F.append({'iid': f'_corpus_L1_{d}', 'gate': 'Q1', 'rule': 'template_share_trace',
                  'severity': ('warn' if share > 0.30 else 'info'),
                  'detail': f'{n}/{len(lst)}={share:.0%} (阈值30%)', **meta})
    atoms = [a for it in items if it['layer'] == 'L1' for a in (it.get('gt_atoms') or [])]
    withref = sum(1 for a in atoms if REF_PAT.search(a))
    pct = f'{withref / len(atoms):.1%}' if atoms else 'n/a(无 L1 题)'
    F.append({'iid': '_corpus_L1', 'gate': 'Q1', 'rule': 'atom_ref_coverage',
              'severity': 'info',
              'detail': f'{withref}/{len(atoms)}={pct} (T2 目标≈100%)', **meta})
    return F
