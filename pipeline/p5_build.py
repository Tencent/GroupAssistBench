#!/usr/bin/env python3
# P5 Build instances (paper 3.2 "building instances"): turn anchors into instances.
# RT1 keeps the original request; RT2 writes a query backwards from anchors; D3 variants
# are built from screened conflict pairs. L1 answer keys are reference atoms; L2 answer
# keys are rubrics of grounding / query-specific / generic / penalty criteria.
# New ids: {gid}-{L}-{4-digit sequence}; ids carry no meaning and are never reused.
#   P5-0 memory selection (code) breadth first: unconsumed anchors first; hub marks come
#                                from event clustering
#   P5-1 query                   RT1 = the original request (src_turn); RT2 = written by the
#                                model, with the question type assigned by code from a
#                                fixed set of nine; the model must return premises
#                                [{text, turn, span}] for the Q8 check
#   P5-2 cut_turn (code)         RT1 = src_turn; RT2 = the median turn in the window
#                                RT2_MIN_SPAN..RT2_MAX_SPAN after the anchors
#   P5-4 answer atoms (code)     L1: one atom per anchor (with #turn); D3-update: the
#                                superseded old value goes to distractor_atoms
#   P5-5 criteria                grounding = template filled from anchor facts (no model);
#                                penalty = C1-C7 trigger scan (code); query_specific /
#                                generic = model (the only free generation)
#   P5-6 pairing (code)          L1 <-> L2 items that share anchor turns are linked
#   self-check                   the deterministic gates Q1/Q2/Q7/Q8 run on every new item;
#                                an error sets status=flagged (nothing is deleted)
# Usage: python3 pipeline/p5_build.py <gid> [--rt1] [--rt2 N|all] [--rt2-d3 N|all] [--rt2-cf N|all] [--no-llm] [--limit N]
#   --rt1           build RT1 items (from candidates + anchors + classify)
#   --rt2 N|all     build RT2 items from the anchor inventory; N is an upper bound,
#                   all = driven by the material (needs a model)
#   --rt2-d3 N|all  build update-type D3 items from the conflict pairs in d3_updates.jsonl
#                   (needs a model); the revised generation is the default, D3_V2=0
#                   switches back to the legacy one
#   --rt2-cf N|all  build arbitration-type D3 items from RESOLVED conflicts in
#                   d3_conflicts.jsonl (unresolved ones are never built; needs a model)
#   --no-llm        skip every model step (L2 items missing query_specific/generic stay draft)
#   --strict-layer  skip positions whose layer is undecided; by default they are built as
#                   L1 and marked provisional
# P5 构题: RT1 保留原始请求, RT2 由锚点反写题面, D3 由筛查过的冲突对构造。
# L1 答案为参考要点, L2 答案为评分标准。新题生成后立即跑确定性闸门, error 记为 flagged。
import difflib, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.llm import MODEL_MAIN, append_jsonl, chat_json, load_jsonl, ParseFail
from pipeline.p2_locate import rundir, load_msgs, build_ctx
from pipeline.p4_classify import (detect_update as p4_detect_update,
                                  substantive_units as p4_substantive_units,
                                  D2_MIN_UNITS as P4_D2_MIN, _norm_fact)
from pipeline import gates
from pipeline.examples import render as render_examples, append_failure

NUM_PAT = re.compile(r'\d+(?:\.\d+)?')

# Nine question types; for RT2 the type is assigned by code in rotation, so the model
# does not drift into one template.
# 九类题型; RT2 的题型由代码轮流指定, 避免模型集中在少数模板上。
QTYPES = ('单点事实召回', '多实体属性映射', '归属指代', '状态更新追踪', '约束遵守',
          '冲突更新消解', '聚合召回', '个性化服务决策', '多人冲突裁决')
# The L2 form of D3. The L1 form asks which value holds (recall); this form asks the
# assistant to carry out a service with the current value. Its grounding points at the
# new value and penalty C5 at reusing the old one.
# D3 的 L2 题型: 不问「以哪个为准」, 而是要求用当前值完成一次服务; grounding 指向新值, C5 惩罚沿用旧值。
D3_L2_QTYPE = '冲突更新后的个性化服务决策'
QTYPE_LAYER = {'个性化服务决策': 'L2', '多人冲突裁决': 'L2', D3_L2_QTYPE: 'L2',
               '冲突演化溯源与建议': 'L2'}  # all other types default to L1
# Query length budget for update-type D3, the same limits as gates.q2_d3_leak (L1 / L2),
# so the writer knows the limit before the gate checks it.
# D3-update 题面字数预算, 与闸门的上限一致, 构题时就先知道上限。
D3_QUERY_BUDGET = {'L1': gates.D3_QUERY_MAX,
                   'L2': int(os.environ.get('D3_QUERY_MAX_L2', '120'))}
QTYPE_OP = {'单点事实召回': 'recall', '多实体属性映射': 'aggregate', '归属指代': 'recall',
            '状态更新追踪': 'recall', '约束遵守': 'recall', '冲突更新消解': 'resolve',
            '聚合召回': 'aggregate', '个性化服务决策': 'recall', '多人冲突裁决': 'resolve',
            D3_L2_QTYPE: 'resolve', '冲突演化溯源与建议': 'resolve', '取值演化链召回': 'resolve'}
# D3 items alternate between the L1 and L2 forms (index = number of items built), 1:1,
# so the same conflict pair can be asked both ways for paired analysis.
# D3 构题时 L1/L2 1:1 轮替, 同一冲突对可得到两种问法。
D3_QTYPES = ('冲突更新消解', D3_L2_QTYPE)

# ---------------- Revised D3: conflict pair + same-matter aggregation ----------------
# Material: besides the conflict pair, take substantive anchors in the window
#   [old anchor - WIN_BEFORE, update turn + WIN_AFTER] that the model judges to concern the
#   same matter; take 2-3 of them, at least one from a speaker other than the updater.
#   Each item then has >= 3 anchors, >= 2 speakers and (L2) >= 2 grounding criteria
#   (coverage_fold, same structure as D2).
# Query: RT2_SYS plus no-cue rules; a hit on gates.D3_HINT_PAT triggers one rewrite, and a
#   remaining hit is an error at the Q2 gate.
# Answer key: no "the old value is void" atom in L1 (unreachable without cue words; kept
#   separately as gt_stale_decl); the old value still goes to distractor_atoms.
# Cutoff: the median turn RT2_MIN_SPAN..RT2_MAX_SPAN after the update, and before the next
#   update of the same matter.
# Conflict pairs must first pass the three-vote screening (r7_pairs_screen.py).
# 新版 D3: 冲突对之外再取同一事项的相关锚点(≥3 个锚点、≥2 位发言人), 题面不含时间线索词,
# 截断点在更新之后、同一事项下一次更新之前。冲突对须先通过三轮投票筛查。
D3_V2 = os.environ.get('D3_V2', '1') != '0'
D3_V2_BOTH_LAYERS = os.environ.get('D3_V2_BOTH_LAYERS', '0') == '1'  # 1 = build both L1 and L2 for every pair (for evaluation); default alternates
# Two forms of revised D3, alternating by conflict pair:
#   service  the query reads like a request from someone who only knows the current
#            arrangement, without cue words; tests implicit detection of the update. Any
#            use of a stale value counts as wrong.
#   trace    the query asks for the whole evolution of the matter before advice or a
#            conclusion; old values are legitimate grounding / answer atoms; tests explicit
#            reconstruction of the chain and tolerates an incomplete chain. The two forms
#            test different abilities and are reported separately.
# 新版 D3 两种题型按冲突对轮替: service 隐式发现更新; trace 显式重建演化链条。
D3_V2_FORMS = tuple(f for f in os.environ.get('D3_V2_FORMS', 'service,trace').split(',') if f in ('service', 'trace'))
# Conflict-pair screening is required (pipeline/r7_pairs_screen.py writes
# <run>/d3_pairs_r7.json): red/yellow pairs without a human true_update verdict are not
# built. D3_R7_REQUIRE=0 is for smoke tests and comparisons only.
# 冲突对筛查为必需; 红/黄且无人工确认的对不出题。D3_R7_REQUIRE=0 仅用于冒烟与对照。
D3_R7_REQUIRE = os.environ.get('D3_R7_REQUIRE', '1') != '0'
# Build only the listed conflict pairs (comma-separated pair_key, e.g. 855->1480); for
# examples or single additions, does not affect the default flow.
# 只构建指定冲突对, 用于样例或单对补题, 不影响默认流程。
D3_PAIRS = frozenset(k for k in os.environ.get('D3_PAIRS', '').split(',') if k)


# Reuse for the revised D3 generation. A conflict pair counts as processed when an item
# was built from it, or when it reached a terminal "no material" outcome; pairs that only
# hit a call error are retried. Pairs blocked by the screening verdict are not counted, so
# a later human verdict can still release them.
# D3 v2 复用: 冲突对已出过题、或已落「取料失败」类最终结论, 即视为已处理; 仅调用出错的重试。
# 被筛查拦下的不算已处理 —— 之后人判放行仍可出题。
_D3V2_TERMINAL = ('P5-1d3', 'P5-1d3span', 'P5-1d3v2')


def _d3v2_log(d, pr, rec):
    rec.setdefault('old_anchor_turn', pr.get('old_anchor_turn'))
    rec['d3v2'] = True
    p = os.path.join(d, 'arbitration_queue.jsonl')
    if rec.get('stage') == 'P5-1d3r7':
        same = lambda r: (r.get('stage') == 'P5-1d3r7' and r.get('update_turn') == rec.get('update_turn')
                          and r.get('old_anchor_turn') == rec.get('old_anchor_turn') and r.get('reason') == rec.get('reason'))
        if os.path.exists(p) and any(same(r) for r in load_jsonl(p)):
            return
    append_jsonl(p, rec)


def d3v2_done(d, items):
    """Returns (processed pair_keys, processed turns known only by update_turn, anchor-turn
    sets of revised D3 items that carry no pair_key).
    """
    keys, turns, legacy = set(), set(), []
    for it in items:
        v = it.get('d3v2')
        if not v:
            continue
        if v.get('pair_key'):
            keys.add(v['pair_key'])
        else:  # items without a pair_key are recognised by their anchor-turn set
            legacy.append({a['turn'] for a in it.get('anchors') or []})
    p = os.path.join(d, 'arbitration_queue.jsonl')
    for r in (load_jsonl(p) if os.path.exists(p) else []):
        st = r.get('stage')
        if r.get('err') or st not in _D3V2_TERMINAL + ('P5-1d3stale',):
            continue
        if st == 'P5-1d3stale' and '弃' not in str(r.get('reason')):
            continue  # the cutoff was moved earlier and the flow continues: not a terminal outcome
        if st in ('P5-1d3', 'P5-1d3span') and not r.get('d3v2'):
            continue  # the legacy path (D3_V2=0) writes these stages too; rows without the marker do not count
        if r.get('old_anchor_turn') is not None:
            keys.add(f"{r['old_anchor_turn']}->{r['update_turn']}")
        else:
            turns.add(r.get('update_turn'))
    return keys, turns, legacy


def d3v2_r7_filter(d, pairs):
    """Filter conflict pairs by d3_pairs_r7.json. Returns (passed pairs, blocked
    [(pair, reason)]). If the file is missing and screening is required -> SystemExit.
    按筛查结论过滤冲突对; 必需模式下文件缺失即退出。
    """
    p = os.path.join(d, 'd3_pairs_r7.json')
    if not os.path.exists(p):
        if D3_R7_REQUIRE:
            raise SystemExit('缺 d3_pairs_r7.json: 先跑 python3 pipeline/r7_pairs_screen.py <run目录名>（冲突对筛查是 D3 v2 出题的前置; '
                             '冒烟可用 D3_R7_REQUIRE=0 跳过）')
        return pairs, []
    verdict = json.load(open(p, encoding='utf-8'))
    ok, blocked = [], []
    for pr in pairs:
        k = f"{pr['old_anchor_turn']}->{pr['update_turn']}"
        v = verdict.get(k)
        if v is None:
            blocked.append((pr, 'r7_unscreened')); continue
        if v.get('human'):
            (ok if v['human'] == 'true_update' else blocked).append(pr if v['human'] == 'true_update' else (pr, f"human:{v['human']}"))
        elif v.get('auto') == 'green':
            ok.append(pr)
        else:
            blocked.append((pr, f"auto:{v.get('auto')}({v.get('majority')})"))
    return ok, blocked
D3_TRACE_L2_QTYPE = '冲突演化溯源与建议'
D3_TRACE_L1_QTYPE = '取值演化链召回'
D3V2_WIN_BEFORE = int(os.environ.get('D3V2_WIN_BEFORE', '120'))
D3V2_WIN_AFTER = int(os.environ.get('D3V2_WIN_AFTER', '40'))
D3V2_N_RELATED = (2, 3)
D3V2_MAX_CANDS = 24

D3V2_RELATE_SYS = """你是数据标注流水线中的「同事项锚点筛选」模块。给定群聊里一次「取值更新」(某事项从旧值改为新值)与一批候选记忆锚点, 判断每条候选是否与这次更新属于**同一事项/同一件事的安排**, 即: 若有人此刻请助手基于该事项当前的取值去办一件事, 这条记忆是**必须一并用到**的事实(同一活动的其他约定、相关人的要求、配套的数量/时间/预算等)。
不算同一事项的: 无关话题、纯寒暄、仅时间上相邻但内容无关、与该事项只有泛泛联系(如都提到「群」)。
只输出严格 JSON: {"related":[候选序号,...],"why":"≤30字"}"""

D3V2_DEHINT_RULE = """
7. **不得使用任何时间线索或版本线索措辞**: 禁止出现「改过/更改/更新/换成/最新/此刻/最终/后来/刚才/之前/原来/新的/旧的/变更/调整后/定下来的」等词, 也不得暗示「这件事前后有过不同说法」。题面就像一个**只知道当前安排**的群成员在提需求 —— 他不知道也不在意曾经改过。答题者必须自己从群历史里分辨哪个取值现行有效。
8. 题面的服务需求必须**自然地**同时依赖「本次更新事项」与「已列出的其他相关记忆」(见锚点清单), 让回答者不得不把这几条记忆一起用上; 但仍不得复述任何锚点的具体值。"""

D3V2_L1_HINT = ('这一档考的是**取值召回与聚合**: 请群成员以「只知道当前安排的人」的口吻, 一次性问清该事项当前的取值以及清单里其他相关记忆的取值'
                '(如「X 的安排和 Y、Z 分别是怎么定的」), 不得暗示曾有过修改, 不得引述任何取值。')

D3V2_TRACE_RULE = """
7. **溯源要求**: 题面要请bot先交代这件事在群里是**怎么一步步定下来的**(先是什么方向、后来为什么改、谁拍的板), 再基于现行安排给出建议或结论; 同时自然带上清单里的其他相关记忆。
8. 题面**不得引述任何取值**, 也不得说明有几个版本、哪一版是最后的——答题者要自己从群历史里把演化链和现行值都找出来。
   反例(违规): 「先说要找画师后来又改成工作室, 现在按工作室…」; 正例: 「新形象制作方这事群里前前后后是怎么定的? 按最后的安排帮我拟一段委托需求。」"""

D3V2_TRACE_L2_HINT = ('这一档考的是**演化溯源 + 用现行值完成一次服务**: 请群成员请bot先按时间顺序交代该事项的演化(每一步是谁、怎么定的), '
                      '再按现行安排产出一件东西或给一次决策, 并自然依赖其他相关记忆。不得引述取值, 不得点明版本数。')
D3V2_TRACE_L1_HINT = ('这一档考的是**取值演化链召回**: 请群成员问清该事项前后各是怎么定的、最终以哪个为准, 以及清单里其他相关记忆的取值; '
                      '不得引述任何取值, 不得点明有几个版本。')

D3V2_CURRENCY_SYS = """你是数据标注流水线中的「取值现行性核查」模块。给定某事项的一次更新(旧值→新值, 发生在更新轮)以及更新轮之后直到提问时刻的群聊原文, 判断: 到提问时刻, 该事项**当前生效的取值是否仍是这个新值**。
若其后群里又改了(明确改口、拍板换方案、新值被放弃), 给出改动发生的轮次与改后的取值; 仅有讨论、疑问、玩笑而未形成新决定的, 视为仍现行。
只输出严格 JSON: {"still_current": true/false, "changed_turn": 整数或null, "latest_value": "≤20字或空", "why": "≤30字"}"""


def d3v2_currency(d, pair, msgs, cut):
    """Is the new value still current at the cutoff? Returns (still_current, changed_turn, note).
    A later change that p4_d3 missed is invisible to d3v2_next_update, so the transcript is
    asked once more; otherwise an item could treat an outdated value as current and mark a
    correct answer as stale.
    新值在截断点是否仍然有效: 对原文再确认一次, 防止 p4_d3 漏检的后续更新导致答案过时。
    """
    ut = pair['update_turn']
    win = [m for m in msgs if ut < m['turn'] < cut and m.get('kind') != 'bot']
    body = '\n'.join(f"#{m['turn']} [{m['who']}] {(m['text'] or '')[:160]}" for m in win)
    ctx = (f"事项「{pair.get('slot')}」: 旧值「{pair.get('old_value')}」(#{pair['old_anchor_turn']}) → "
           f"新值「{pair.get('new_value')}」(#{ut})\n提问时刻: #{cut}\n===更新轮之后的群聊===\n{body or '(无人类发言)'}")
    v = chat_json(d, 'p5_d3v2cur', D3V2_CURRENCY_SYS, ctx, max_tokens=200)
    ct = v.get('changed_turn')
    ct = int(ct) if isinstance(ct, (int, float)) or (isinstance(ct, str) and ct.isdigit()) else None
    return bool(v.get('still_current', True)), ct, f"{v.get('latest_value', '')}|{v.get('why', '')}"[:80]


def d3v2_dedupe_pairs(pairs):
    """Keep one of duplicate events (same old anchor + same update turn, or same old anchor +
    same new-value prefix); overlapping p4_d3 windows and similar slot names repeat events.
    去除重复事件。
    """
    out, seen = [], set()
    for p in sorted(pairs, key=lambda x: x['update_turn']):
        k1 = (p['old_anchor_turn'], p['update_turn'])
        k2 = (p['old_anchor_turn'], (p.get('new_value') or '')[:6])
        if k1 in seen or k2 in seen:
            continue
        seen.add(k1); seen.add(k2)
        out.append(p)
    return out


def d3v2_next_update(pair, pairs):
    """The next update turn of the same matter after update_turn (the cutoff must precede it),
    or None.
    """
    nxt = [q['update_turn'] for q in pairs if q is not pair and q['update_turn'] > pair['update_turn']
           and (q.get('old_anchor_turn') in (pair['update_turn'], pair['old_anchor_turn'])
                or (q.get('slot') and pair.get('slot') and _bigrams(q['slot']) & _bigrams(pair['slot'])))]
    return min(nxt) if nxt else None


def d3v2_cut(pair, msgs, pairs):
    ut = pair['update_turn']
    hi = ut + RT2_MAX_SPAN
    nxt = d3v2_next_update(pair, pairs)
    if nxt:
        hi = min(hi, nxt - 1)
    cands = [t for t in sorted(m['turn'] for m in msgs) if ut + RT2_MIN_SPAN <= t <= hi]
    return cands[len(cands) // 2] if cands else None


def d3v2_candidates(pair, inv, cut, core_turns):
    lo, hi = pair['old_anchor_turn'] - D3V2_WIN_BEFORE, min(pair['update_turn'] + D3V2_WIN_AFTER, cut - 1)
    pool = [a for a in inv if a.get('type') != '人设' and not a.get('bot_sourced')
            and a['turn'] not in core_turns and lo <= a['turn'] <= hi]
    pool = p4_substantive_units(pool)  # same-turn and near-duplicate folding, as in P4
    pool.sort(key=lambda a: abs(a['turn'] - pair['update_turn']))
    return pool[:D3V2_MAX_CANDS]


def d3v2_relate(d, pair, cands, who_of):
    lst = '\n'.join(f"{i + 1}. #{a['turn']} [{who_of(a['turn'])}] {a['fact']}" for i, a in enumerate(cands))
    ctx = (f"本次更新: 事项「{pair.get('slot')}」 旧值「{pair.get('old_value')}」(#{pair['old_anchor_turn']}) → "
           f"新值「{pair.get('new_value')}」(#{pair['update_turn']})\n候选锚点:\n{lst}")
    v = chat_json(d, 'p5_d3v2rel', D3V2_RELATE_SYS, ctx, max_tokens=300)
    idx = [int(i) for i in (v.get('related') or []) if str(i).isdigit() and 1 <= int(i) <= len(cands)]
    return [cands[i - 1] for i in idx], str(v.get('why', ''))[:60]


def d3v2_choose(related, upd_who, who_of):
    """Take 2-3 anchors by weight, at least one from a speaker other than the updater (the
    >= 2 speakers invariant).
    按承重度取 2–3 条, 至少 1 条来自更新者之外的发言人。
    """
    if len(related) < D3V2_N_RELATED[0]:
        return None
    others = [a for a in related if who_of(a['turn']) != upd_who]
    if not others:
        return None
    ranked = sorted(related, key=lambda a: -anchor_weight(a, related))
    pick = [others[0]] + [a for a in ranked if a is not others[0]]
    return sorted(pick[:D3V2_N_RELATED[1]], key=lambda a: a['turn'])


def d3v2_qtype(layer, form):
    if form == 'trace':
        return D3_TRACE_L2_QTYPE if layer == 'L2' else D3_TRACE_L1_QTYPE
    return D3_L2_QTYPE if layer == 'L2' else '冲突更新消解'


def d3v2_query(d, msgs, anchors, related, layer, gid, form='service'):
    """RT2_SYS + type hint + (service: no-cue rules | trace: evolution rules); rewritten once
    on cue words (service only) or when over budget.
    """
    qtype = d3v2_qtype(layer, form)
    lo = max(0, min(a['turn'] for a in anchors) - 15)
    hi = max(a['turn'] for a in anchors) + 5
    win = [m for m in msgs if lo <= m['turn'] <= hi]
    if form == 'trace':
        hint = D3V2_TRACE_L2_HINT if layer == 'L2' else D3V2_TRACE_L1_HINT
    else:
        hint = QTYPE_HINT[qtype] if layer == 'L2' else D3V2_L1_HINT
    budget = D3_QUERY_BUDGET[layer]
    ctx = ('===原文窗口===\n' + '\n'.join(f"#{m['turn']} [{m['who']}] {(m['text'] or '')[:200]}" for m in win) +
           '\n===锚点(答案所在, 题面不得复述其值)===\n' +
           '\n'.join(f"#{a['turn']} {a['fact']}" for a in anchors) +
           '\n===其中「其他相关记忆」(服务需求须自然依赖它们)===\n' +
           '\n'.join(f"#{a['turn']} {a['fact']}" for a in related) +
           f'\n===指定题型===\n{qtype}\n{hint}'
           f'\n===字数预算===\n题面全文(含@bot与标点)不超过 {budget} 字——这是闸门硬上限, 超限整题作废。')
    sys_p = RT2_SYS + (D3V2_TRACE_RULE if form == 'trace' else D3V2_DEHINT_RULE) + render_examples('rt2_query', exclude_gid=gid)
    v = chat_json(d, 'p5_rt2q', sys_p, ctx, max_tokens=700)
    q = v.get('query') or ''
    hits = gates.D3_HINT_PAT.findall(q) if form == 'service' else []
    if hits or len(q) > budget:
        why = (f'含时间/版本线索词 {hits}' if hits else '') + (f' 超预算({len(q)}>{budget})' if len(q) > budget else '')
        try:
            v2 = chat_json(d, 'p5_rt2q', sys_p,
                           ctx + f'\n===重写===\n上一稿: {q}\n问题: {why}。重写: 删除所有线索词, 不暗示曾修改, '
                                 f'不超过 {budget} 字, premises 照常输出。', max_tokens=700)
            if v2.get('query'):
                v = v2
        except ParseFail:
            pass
    return v


# ---------------- id allocation: ids carry no meaning, sequence numbers are never reused ----------------

def next_iid(items_existing, gid, layer):
    used = [int(m.group(1)) for it in items_existing
            for m in [re.match(rf'{re.escape(gid)}-{layer}-(\d{{4}})$', it['iid'])] if m]
    return f'{gid}-{layer}-{(max(used) + 1) if used else 1:04d}'


# ---------------- P5-4 answer atoms (L1, code) ----------------

def build_gt_atoms(anchors, conflict, stale_decl=True):
    """One atom per anchor (each fact passed the P3-4 check and carries a #turn citation).
    D3-update: the superseded old-value anchor moves to distractor_atoms.
    D3 also gets a "the earlier value is no longer valid" atom, because a correct D3 answer
    states two things: what the value is now, and that the earlier one is void. The old value
    stays in distractor_atoms as well; the roles differ: the atom asks the answer to say it is
    void, the distractor that it must not be used as current.
    每个锚点一条要点。D3: 旧值进干扰项, 另加一条「旧值已失效」要点。
    """
    old_turns = {h['old_anchor_turn'] for h in (conflict or [])}
    atoms, distractors = [], []
    stale = []
    for a in anchors:
        s = f"{a['fact']}(#{a['turn']})"
        if a['turn'] in old_turns:
            distractors.append(s)
            stale.append(a)
        else:
            atoms.append(s)
    for h in (conflict or []) if stale_decl else []:  # not added for revised D3: unreachable without cue words in the query
        a = next((x for x in stale if x['turn'] == h['old_anchor_turn']), None)
        if a is None:
            continue
        atoms.append(f"此前的「{a['fact']}」(#{a['turn']})已被 #{h['update_turn']} 覆盖, "
                     f"不再作为当前取值")
    return atoms, distractors


# ---------------- P5-5 criteria (L2) ----------------

BG_ANCHOR_TYPES = {'人设'}  # background anchors: lowest weight


def _bigrams(t):
    t = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', t or '')
    return {t[i:i + 2] for i in range(len(t) - 1)}


def anchor_weight(a, anchors):
    """Anchor weight (code, no model). Three deterministic signals:
      recency      the point at issue usually sits close to the request, background earlier;
                   normalised to 0-1, the largest weight
      specificity  facts with >= 2-digit values or dates are easier to check (C1 relies on it)
      type         persona entries are background and get the lowest weight
    Without weighting, taking anchors in turn order would always pick the earliest,
    background ones and leave out the point at issue.
    锚点承重度: 近因、具体性、类型三项确定性信号; 避免总是选中最早的背景锚点。
    """
    turns = [x['turn'] for x in anchors] or [0]
    lo, hi = min(turns), max(turns)
    recency = (a['turn'] - lo) / (hi - lo) if hi > lo else 0.5
    concrete = 0.3 if [n for n in NUM_PAT.findall(a.get('fact') or '') if len(n) >= 2] else 0.0
    bg = -0.5 if a.get('type') in BG_ANCHOR_TYPES else 0.0
    return recency + concrete + bg


FOLD_MAX_FACTS = 6  # maximum facts required by folding in total (lower-weight anchors beyond this are left out)
FOLD_CHUNK = 2  # facts per folded criterion
FOLD_MAX_CRITS = 3  # maximum number of folded criteria


def grounding_criteria(anchors, gid, iid, op_type=None):
    """Grounding criteria. 1-4 of them; points: the first (core anchor) 3/must, the others
    2/must (beyond 4 must, demoted to bonus 1). Selection and points follow anchor_weight,
    not list order.
    For op_type=aggregate the facts are folded into coverage criteria: at most 3 criteria,
    at most 2 facts each, at most 6 facts in total, and a criterion is met only when all its
    facts are covered. Judges score each criterion as met / not met, so packing many facts
    into one criterion would leave no partial credit; several small criteria let partial
    credit come from the number met, and the heaviest anchors go into the 3-point ones.
    grounding 判据: 按承重度选取与配分; 聚合题折叠为至多 3 条、每条至多 2 个事实的覆盖型判据。
    """
    # Fold near-duplicates before selecting, using the same substantive_units rule as the
    # difficulty, so one fact is not scored twice.
    # 选料前先折叠近重复, 与难度判定用同一套规则, 避免同一事实重复计分。
    anchors = p4_substantive_units(anchors) or anchors
    ranked = sorted(anchors, key=lambda a: -anchor_weight(a, anchors))
    if op_type == 'aggregate' and len(ranked) >= 2:
        sel = ranked[:FOLD_MAX_FACTS]
        chunks = [sel[i:i + FOLD_CHUNK] for i in range(0, len(sel), FOLD_CHUNK)][:FOLD_MAX_CRITS]
        out = []
        for i, ch in enumerate(chunks):
            items = '; '.join(f"{a['fact']}(#{a['turn']})" for a in ch)
            out.append({'cid': f'{iid}-c{i + 1}',
                        'check': f'回复需覆盖以下事实(全部覆盖才算满足): {items}',
                        'kind': 'grounding', 'polarity': 'must', 'points': 3 if i == 0 else 2,
                        'source': 'anchors:' + ','.join(f"#{a['turn']}" for a in ch),
                        'by': 'rule:coverage_fold', 'coverage_n': len(ch),
                        'fold_idx': i + 1, 'fold_of': len(chunks)})
        return out
    out = []
    for i, a in enumerate(ranked[:4]):
        pts, pol = (3, 'must') if i == 0 else (2, 'must')
        if i >= 3:  # keep room for the must <= 4 limit
            pts, pol = 1, 'bonus'
        out.append({'cid': f'{iid}-c{len(out) + 1}',
                    'check': f"回复需用到「{a['fact']}」(#{a['turn']})",
                    'kind': 'grounding', 'polarity': pol, 'points': pts,
                    'source': f"anchor#{a['turn']}", 'by': 'rule:template_fill'})
    return out


def penalty_criteria(anchors, profile, conflict, who, cut, iid, start_n, grounded=(),
                     who_of=None, rel_ctx=None):
    """C1-C7 trigger scan (code); every criterion cites an evidence turn; the top <= 3 by
    |weight| are kept. Weights: C4=-3 C1=-2 C3=-2 C5=-1 C2=-1.
    Penalty criteria describe the mistake itself ("misremembers X", "contradicts Y") rather
    than a prohibition ("does not misremember X"), so that met = the answer made the mistake,
    whatever reading the judge takes.
    C1 must not duplicate grounding: `grounded` is the set of turns already used by
    grounding, and C1 picks a numeric anchor outside it, so one fact is not rewarded and
    penalised at the same time.
    扣分项扫描: 每条都引用证据轮次, 按权重取前 3。措辞陈述错误本身, 使「met = 犯了这个错」;
    C1 不与 grounding 取同一锚点。
    """
    cands = []
    num_anchor = next((a for a in anchors
                       if NUM_PAT.search(a.get('fact') or '') and a['turn'] not in grounded), None)
    if num_anchor:
        cands.append(('C1', -2, f"记错或篡改了「{num_anchor['fact']}」(#{num_anchor['turn']})",
                      num_anchor['turn']))
    for h in (conflict or [])[:1]:
        cands.append(('C5', -1,
                      f"按 #{h['old_anchor_turn']} 的旧值回答, 未采用 #{h['update_turn']} 更新后的值",
                      h['update_turn']))
    prof_upto = [p for p in profile if isinstance(p.get('turn'), int) and p['turn'] <= (cut or 10 ** 9)]
    # C3 is taken only from explicit agreements made in the group: slot = constraint, not
    # inherited, turn > 0. A persona biography from the profile is not an agreement. If none
    # exists, no C3 is generated.
    # C3 只取群内明确提出的约定(非继承的约束条目), 取不到就不生成。
    persona = next((p for p in prof_upto
                    if p.get('slot') == '约束' and p.get('turn', 0) > 0
                    and p.get('verified') != 'inherited'
                    and not str(p.get('by', '')).startswith('inherited')), None)
    if persona:
        cands.append(('C3', -2, f"违背了 #{persona['turn']} 约定的「{persona['fact']}」", persona['turn']))
    state = next((p for p in reversed(prof_upto) if p['slot'] == '状态'
                  and (not who or p['member'].startswith(who.split('（')[0]))
                  and (cut is None or cut - p['turn'] <= 150)), None)  # recency filter: stale states do not instantiate C4
    if state:
        cands.append(('C4', -3, f"把 {state['member']} 的当前状态「{state['fact']}」(#{state['turn']}) "
                                f"误读、错定性或归错因", state['turn']))
    plan = next((p for p in prof_upto if p['slot'] == '既有方案'), None)
    if plan:
        cands.append(('C2', -1, f"与 #{plan['turn']} 既有方案「{plan['fact']}」矛盾", plan['turn']))
    # C2/C3 must be relevant to this item: the cited fact must share at least one bigram with
    # the query or the grounding. Distance is not used, because rules stated long ago can
    # still apply.
    # C2/C3 须与本题相关(与题面或 grounding 至少共享一个 bigram); 不用距离限制, 早期群规可能仍然有效。
    if rel_ctx:
        keep = []
        for cl, pts, chk, ev in cands:
            if cl in ('C2', 'C3'):
                m = re.search(r'「(.+?)」', chk)
                if m and not (_bigrams(m.group(1)) & rel_ctx):
                    continue  # no lexical overlap with this item: not generated
            keep.append((cl, pts, chk, ev))
        cands = keep
    # ---- Fallback chain (only when no profile-based trigger is left) ----
    # Runs after the relevance check, so "all removed" and "none triggered" take the same path.
    #   C6 does not depend on the profile: when the anchors come from >= 2 speakers,
    #      attributing a fact to the wrong person is a real possible mistake.
    #   C1 may share an anchor with grounding when nothing else is available: with criteria
    #      worded as mistakes they are complementary, not duplicates -
    #        answer omits X -> grounding false, C1 false (not wrong, just missing)
    #        answer gets X wrong -> grounding false, C1 true (misremembered)
    #      so C1 separates omission from distortion. Still a fallback only, to avoid
    #      systematic double weighting.
    # 兜底链(相关性检查之后): C6 张冠李戴; 无其他扣分项时 C1 允许与 grounding 同源, 区分「遗漏」与「记错」。
    if not cands:
        fb = next((a for a in anchors if NUM_PAT.search(a.get('fact') or '')), None)
        if fb:
            cands.append(('C1', -2, f"记错或篡改了「{fb['fact']}」(#{fb['turn']})", fb['turn']))
    speakers = {who_of(a['turn']) for a in anchors} - {None, ''} if who_of else set()
    if not cands and len(speakers) >= 2:
        whos = sorted(speakers)[:3]
        cands.append(('C6', -2, f"把 {'、'.join(whos)} 之间的信息张冠李戴, 归错到别人身上",
                      min(a['turn'] for a in anchors)))
    # C7: the failure mode of two-person groups, where C6 cannot apply. Generic "pitfall"
    # templates are not used because they cannot be verified against the conversation. C7
    # uses the one verifiable mistake: the assistant proposed something the user never
    # confirmed, and the answer treats it as decided. The evidence turn is the latest bot
    # anchor of the item, which a judge can check verbatim.
    # C7: 两人群的失败模式——把助手提议过、用户并未确认的方案当成已定。证据为最晚的 bot 锚点。
    if not cands:
        bot_ts = [a['turn'] for a in anchors if a.get('bot_sourced')]
        if bot_ts:
            ev = max(bot_ts)
            cands.append(('C7', -2, f"把 #{ev} 里bot提议过、但群成员并未确认采纳的内容, "
                                    f"当成已经定下来的事实来回答", ev))
    # Last resort: C1 without the numeric requirement. A criterion pointing at a non-numeric
    # fact (still with #turn and the original text) is better than no penalty at all.
    # 最后一档: C1 不再要求含数值。
    if not cands and anchors:
        a0 = max(anchors, key=lambda a: anchor_weight(a, anchors))
        cands.append(('C1', -2, f"记错或篡改了「{a0['fact']}」(#{a0['turn']})", a0['turn']))
    cands.sort(key=lambda x: x[1])  # larger absolute weight first
    out = []
    for cluster, pts, check, ev in cands[:3]:
        out.append({'cid': f'{iid}-c{start_n + len(out) + 1}', 'check': check,
                    'kind': 'penalty', 'polarity': None, 'points': pts,
                    'source': f'feedback({cluster})', 'by': 'rule:cluster_trigger'})
    return out


def grounded_turns(crits):
    """Anchor turns counted as used by grounding when C1 checks for duplication - both
    per-anchor and folded criteria. Otherwise one wrong value would earn +3 under grounding
    and -2 under C1 at the same time.
    When the profile holds only inherited persona entries, C2/C3/C4 cannot trigger and C1
    avoids grounding, so aggregate items may get no penalty; the gates then report a warn
    (gates.q1_item) rather than passing it silently.
    C1 判同源时视为已被 grounding 占用的锚点轮次(逐条型与折叠型都算)。
    """
    return {int(m.group(1)) for c in crits
            for m in re.finditer(r'#(\d+)', c.get('source') or '')}


def enforce_caps(crits, max_pos=6, max_pen=3, max_total=8, max_must=4):
    """Count limits: positive criteria 3-6, penalty <= 3, must <= 4, <= 8 in total.
    Must run AFTER query_specific has been appended; truncating earlier would keep the
    wrong entries and drop the right ones.
      penalty   keep <= 3, most negative first
      positive  grounding and query_specific kept in alternation, each by points
                descending, so neither kind is removed entirely
      must      beyond 4, the lowest are demoted to bonus (1 point), not dropped
      cid       renumbered consecutively
    条数上限; 必须在追加 query_specific 之后执行。
    """
    pos = [c for c in crits if int(c['points']) > 0]
    pen = sorted([c for c in crits if int(c['points']) < 0], key=lambda c: int(c['points']))[:max_pen]
    g = sorted([c for c in pos if c['kind'] == 'grounding'], key=lambda c: -int(c['points']))
    q = sorted([c for c in pos if c['kind'] != 'grounding'], key=lambda c: -int(c['points']))
    room = max(0, min(max_pos, max_total - len(pen)))
    keep, take_g = [], True
    while len(keep) < room and (g or q):
        src = g if (take_g and g) or not q else q
        keep.append(src.pop(0))
        take_g = not take_g
    musts = [c for c in keep if c.get('polarity') == 'must']
    for c in sorted(musts, key=lambda c: int(c['points']))[:max(0, len(musts) - max_must)]:
        c['polarity'], c['points'] = 'bonus', 1
    out = keep + pen
    for n, c in enumerate(out, 1):
        c['cid'] = re.sub(r'-c\d+$', f'-c{n}', c['cid'])
    return out


QS_SYS = """你是数据标注流水线中的「query_specific 条目」生成器。给定一道 L2 评测题(题面+锚点+已有判据), 补充 1–3 条本题特有的正分判定条目。这是判据生成中唯一的自由生成步, 输出会被结构 lint 与无记忆探针闸门校验。

# 硬判据
- 每条是「可逐条判定的一句话」, 落到可执行/可核对的行为, 不写元描述
- 禁用措辞: 「必须明确提及」「必须指出」「必须复述」(这是给判分者的指令, 不是对回答的要求)
- 不与已有 grounding/penalty 条目语义重复; 广泛适用于任何题的通用要求不要写(那是 generic 的活)
- 个性化钩子: 条目应要求「说得出为什么适合此人/此群」, 而非笼统好建议

# 输出严格 JSON
{"criteria":[{"check":"...","polarity":"must|bonus","reason":"≤15字"}]}"""


def qs_criteria(d, item, iid, start_n):
    ctx = (f"题面: {item['query']}\n锚点: " +
           '; '.join(f"{a['fact']}(#{a['turn']})" for a in item['anchors']) +
           '\n已有判据: ' + '; '.join(c['check'] for c in item['criteria']))
    v = chat_json(d, 'p5_qs', QS_SYS + render_examples('query_specific', exclude_gid=item['gid']),
                  ctx, max_tokens=500)
    out = []
    for c in (v.get('criteria') or [])[:3]:
        pol = c.get('polarity') if c.get('polarity') in ('must', 'bonus') else 'bonus'
        out.append({'cid': f'{iid}-c{start_n + len(out) + 1}', 'check': str(c.get('check', ''))[:120],
                    'kind': 'generic', 'polarity': pol, 'points': 2 if pol == 'must' else 1,
                    'source': 'expert(model提案,极性确认权在人)', 'by': f'model:main({MODEL_MAIN})',
                    'provisional': True})
    return out


# ---------------- RT2 query writing (main model) ----------------
# RT2 cutoffs must leave a real memory span: the cutoff is searched from RT2_MIN_SPAN
# turns after the anchors, and no update of the anchor may occur in that interval
# (otherwise the anchor is void and the answer would be wrong). Without a valid position
# the anchor is skipped.
# RT2 截断点须留出真实记忆跨度: 从锚点之后至少 RT2_MIN_SPAN 轮起找, 区间内锚点不能被更新; 找不到就跳过。
RT2_MIN_SPAN = int(os.environ.get('RT2_MIN_SPAN', '20'))
RT2_MAX_SPAN = int(os.environ.get('RT2_MAX_SPAN', '200'))
# RT2 builds anchor groups: the question type decides how many anchors it needs
# (aggregation and multi-entity types need several), neighbours within the event are
# taken, and the difficulty follows the same substantive_units rule as RT1.
# RT2 按题型决定锚点数量, 就近取组; 难度与 RT1 使用同一套判据。
QTYPE_GROUP_N = {'聚合召回': 4, '多实体属性映射': 3, '多人冲突裁决': 3,
                 '个性化服务决策': 2, '冲突更新消解': 2, '状态更新追踪': 2}

# RT2 may reuse anchors, with two limits:
#   1. one anchor appears in at most RT2_ANCHOR_MAX_USE RT2 items; otherwise a popular
#      anchor would enter many items, which breaks independence between items;
#   2. a new group's anchor-set Jaccard with every existing group must be below
#      RT2_GROUP_MAX_JACCARD ({A,B} and {A,C} centred on A are near-duplicates).
# Ordering stays breadth first: unconsumed first, then least used in RT2, then least
# cited by RT1.
# RT2 允许有上限地复用锚点: 单锚点使用次数与组合相似度都有上限; 排序仍为广度优先。
RT2_ANCHOR_MAX_USE = int(os.environ.get('RT2_ANCHOR_MAX_USE', '3'))
RT2_GROUP_MAX_JACCARD = float(os.environ.get('RT2_GROUP_MAX_JACCARD', '0.5'))


def rt2_pool(inv, used_count):
    """RT2 pool: all anchors except persona / bot_sourced, sorted breadth first, without those
    that reached the reuse limit.
    """
    pool = [a for a in inv if a.get('type') != '人设' and not a.get('bot_sourced')
            and used_count.get(a['turn'], 0) < RT2_ANCHOR_MAX_USE]
    pool.sort(key=lambda a: (bool(a.get('consumed_by_items')),  # unconsumed first
                             used_count.get(a['turn'], 0),  # least used in RT2 first
                             len(a.get('used_by_queries') or []),  # least cited by RT1 first
                             a['turn']))
    return pool


# RT2's dedicated D3 path. The regular RT2 path cannot produce D3: its difficulty comes
# from unit counts (D1/D2 only) and its grouping knows nothing about conflicts. D3 needs a
# conflict pair (the superseded old-value anchor + the new value), supplied by p4_d3.py in
# d3_updates.jsonl. Unlike the regular path, anchors need not be unconsumed: the core
# anchor of a D3 item is often one RT1 already uses.
# RT2 的 D3 专用路径: 冲突对来自 p4_d3.py; 不要求锚点未被使用。
def load_d3_pairs(d):
    """Read p4_d3's output; only events matched to an old-value anchor (without one there is no D3)."""
    p = os.path.join(d, 'd3_updates.jsonl')
    if not os.path.exists(p):
        return []
    out = []
    for r in load_jsonl(p):
        if r.get('old_anchor_turn') is None:
            continue
        out.append(r)
    return out


def d3_anchor_group(pair, inv, msgs):
    """Conflict pair -> anchor group: [old-value anchor (-> distractor), new-value anchor (answer)].
    The new-value anchor is looked up in the inventory at update_turn; if absent it is built
    from the event's new_value, with the update sentence as span so that Q7 can verify it.
    冲突对 → 锚点组: 旧值锚(干扰项) + 新值锚(答案)。
    """
    old_a = next((a for a in inv if a['turn'] == pair['old_anchor_turn']
                  and a['fact'] == pair.get('old_anchor_fact')), None)
    if old_a is None:
        old_a = next((a for a in inv if a['turn'] == pair['old_anchor_turn']), None)
    if old_a is None:
        return None
    ut = pair['update_turn']
    new_a = next((a for a in inv if a['turn'] == ut), None)
    if new_a is None:
        txt = next((m.get('text') or '' for m in msgs if m['turn'] == ut), '')
        nv = (pair.get('new_value') or '').strip()
        span = nv if nv and nv in txt else txt[:20]
        if not span:
            return None
        new_a = {'turn': ut, 'fact': f"{pair.get('slot')}已更新为{nv}", 'type': '状态',
                 'by': 'rule:d3_new_value', 'evidence_span': span}
    return [old_a, new_a]


def anchor_views(group, bot_turns):
    """Anchors -> item.anchors view, shared by the D3 and RT2 branches. bot_sourced is set
    from the transcript (kind == 'bot'), which is authoritative whether or not the anchor
    is in the inventory.
    锚点 → item.anchors; bot_sourced 按原文的 kind 判定。
    """
    return [{'turn': x['turn'], 'fact': x['fact'], 'type': x.get('type'),
             'by': x.get('by'), 'evidence_span': x.get('evidence_span'),
             'bot_sourced': bool(x.get('bot_sourced')) or x['turn'] in bot_turns}
            for x in group]


def rt2_group_ok(group, seen_groups):
    """Group dedup: reject when the Jaccard with any existing group is too high."""
    g = {a['turn'] for a in group}
    for h in seen_groups:
        u = len(g | h)
        if u and len(g & h) / u >= RT2_GROUP_MAX_JACCARD:
            return False
    return True


def rt2_pick_group(anchor, pool, qtype):
    """Pick an RT2 anchor group: start from `anchor` and add the nearest anchors within
    RT2_MAX_SPAN until the type's count is reached. Only distinct turns are taken (fields of
    one sentence are not independent memories).
    """
    n = QTYPE_GROUP_N.get(qtype, 1)
    if n <= 1:
        return [anchor]
    used = {anchor['turn']}
    grp = [anchor]
    for b in sorted(pool, key=lambda x: abs(x['turn'] - anchor['turn'])):
        if len(grp) >= n:
            break
        if b['turn'] in used or abs(b['turn'] - anchor['turn']) > RT2_MAX_SPAN:
            continue
        used.add(b['turn'])
        grp.append(b)
    return sorted(grp, key=lambda x: x['turn'])


def rt2_cut_turn(anchor_turn, msgs, detect_stale=None):
    """Pick an RT2 cutoff with a real memory span. None means this anchor position cannot be built."""
    turns = sorted(m['turn'] for m in msgs)
    if not turns:
        return None
    lo, hi = anchor_turn + RT2_MIN_SPAN, anchor_turn + RT2_MAX_SPAN
    cands = [t for t in turns if lo <= t <= hi]
    if not cands:
        return None  # anchor too close to the end of the timeline: span too short, not built
    cut = cands[len(cands) // 2]  # take the median of the interval, not always the lower bound
    if detect_stale and detect_stale(anchor_turn, cut):
        return None  # the anchor is updated within the interval: the answer would be wrong
    return cut


RT2_SYS = """你是数据标注流水线中的「锚点构题」模块。给定群聊原文窗口与指定的锚点事实, 反写一道该群成员会真实提出的评测题面。输出会被程序校验: premises 里每条前提的 span 必须是所指 turn 原文的连续子串, 校验失败整题作废。

# 硬约束
1. 题型已由系统指定(见输入), 不得改变考察操作
2. 题面不得出现锚点事实的具体值(数值/人名/结论)——那是答案; 但须含能唯一定位该记忆的场景要素
3. 范围引导只给数量与形式(如「请分两点说明」), 不给维度名(「从情绪和行为两方面」违规)
4. 语气口语化, 以群成员向 @bot 提问的形态书写
5. **时间边界**: 提问发生在「此刻」, 题面只能引用**已经发生过**的事。严禁出现
   「后来 / 最后 / 后面 / 之后 / 随后 / 事后」这类事后视角措辞, 也不得陈述提问时刻
   尚未发生的事(「后面我还问过你」「他后面又补了一句」)。这类题在提问时刻没有答案存在。
6. **不得把答案铺在题面里**(冲突更新题尤其容易犯)。题面可以点出「这件事前后说法不一致」,
   但**不能把两个说法的内容都引述出来** —— 一旦两个取值都写进题面, 答题者不需要任何记忆,
   只按「以后说的为准」就能答对, 这道题就不再考记忆了。
   反例(违规): 「你前面说"每周五提醒从X开始", 后面又给了"首次提醒Y(周五)", 以哪个为准?」
   正例: 「关于每周五提醒的首次日期, 群里前后有过不同说法。按当前设定应该是哪个? 依据是哪一轮?」
   判断标准: 把题面单独拿给一个**没看过群聊**的人, 他不应该能推出答案。

# 自检(输出前逐条自答)
- 题面里出现了锚点事实的任何一个具体值吗? 出现即重写
- 题面陈述的每个前提, 我能指出它在哪一轮吗? 不能即删除该前提
- 引导语里有没有维度名? 有即删
- 题面里有「后来/最后/后面/之后/随后/事后」吗? 有即改写成「此刻已知」的表述
- 把题面单独给一个没看过群聊的人, 他能推出答案吗? 能即说明答案铺在题面里了, 删掉引述的取值

# 输出严格 JSON
{"query":"@bot ...","premises":[{"text":"前提一句话","turn":123,"span":"该turn原文连续子串(≤20字)"}]}"""


# Type-specific hints, written only for the two D3 types: they share the same anchors
# (old + new value) but must ask different things, and the type name alone does not
# separate them.
# 题型补充说明, 只给两个 D3 题型: 它们共用同一批锚点却要问不同的东西。
QTYPE_HINT = {
    '冲突更新消解':
        '这一档考的是**取值判定**: 问「按当前设定应该是哪个」以及依据在哪一轮。'
        '不要把两个取值的内容都写进题面。',
    D3_L2_QTYPE:
        '这一档考的是**用当前值去完成一次服务**, 不是问「以哪个为准」。'
        '题面应当请bot基于该槽位现在生效的取值, 实际产出一件东西或给一次决策'
        '(如拟一份通知/排一次安排/给一条可执行建议), 并交代清楚服务场景与想要的交付形式。'
        '**不得**在题面里问「前后说法哪个为准」, 也不得引述任何一个取值的内容 ——'
        '答题者必须自己从群历史里取到现行值才能完成这次服务; 沿用作废的旧值即为失败。',
}


def rt2_query(d, msgs, profile, anchors, qtype, gid=None, q_budget=None):
    lo = max(0, min(a['turn'] for a in anchors) - 15)
    hi = max(a['turn'] for a in anchors) + 5
    win = [m for m in msgs if lo <= m['turn'] <= hi]
    hint = QTYPE_HINT.get(qtype)
    ctx = ('===原文窗口===\n' + '\n'.join(f"#{m['turn']} [{m['who']}] {(m['text'] or '')[:200]}" for m in win) +
           '\n===锚点(答案所在, 题面不得复述其值)===\n' +
           '\n'.join(f"#{a['turn']} {a['fact']}" for a in anchors) +
           f'\n===指定题型===\n{qtype}' + (f'\n{hint}' if hint else ''))
    if q_budget:
        ctx += (f'\n===字数预算===\n题面全文(含@bot与标点)不超过 {q_budget} 字——这是闸门硬上限, '
                '超限整题作废。指认历史用转指(「开群时那版」「刚才那段」), 不复述其内容。')
    # Only negatives are shown, no reference queries: positive queries get copied verbatim.
    sys_p = RT2_SYS + render_examples('rt2_query', exclude_gid=gid)
    v = chat_json(d, 'p5_rt2q', sys_p, ctx, max_tokens=700)
    q = v.get('query') or ''
    if q_budget and len(q) > q_budget:  # over budget: one compressing rewrite; if still over or failed, the draft goes to the gates
        try:
            v2 = chat_json(d, 'p5_rt2q', sys_p,
                           ctx + f'\n===压缩重写===\n上一稿题面 {len(q)} 字超预算: {q}\n'
                                 f'重写为不超过 {q_budget} 字: 考察点与前提不变, '
                                 '不得新增任何取值引述, premises 照常输出。',
                           max_tokens=700)
            q2 = v2.get('query') or ''
            if q2 and len(q2) < len(q):
                v = v2
        except ParseFail:
            pass
    return v


# ---------------- item assembly and self-check ----------------

def make_item(gid, iid, layer, cand, anchors, cls, route, extra=None, track='jsonl'):
    # track comes from src.track in messages.jsonl (via the caller); criteria_origin follows
    # the contract: rubrics generated by the pipeline are always autogen.
    it = {'iid': iid, 'legacy_id': None, 'gid': gid, 'layer': layer,
          'difficulty': cls['difficulty']['v'], 'op_type': cls['op_type']['v'],
          'conflict_kind': cls['difficulty'].get('conflict_kind'),
          'track': track, 'query_origin': 'native' if route == 'RT1' else 'authored',
          'criteria_origin': 'autogen' if layer == 'L2' else 'annotation',
          'query_form': cand.get('query_form', 'request') if route == 'RT1' else 'request',
          'query': cand['query'], 'query_prev': None, 'query_treatment': 'raw',
          'src_turn': cand.get('src_turn') if route == 'RT1' else None,
          'cut_turn': (cand['src_turn'] if route == 'RT1'
                       else (extra or {}).get('cut_turn') or max(a['turn'] for a in anchors) + 1),
          # bot_sourced is stored on each item anchor so that downstream checks can see it.
          'anchors': [{'turn': a['turn'], 'fact': a['fact'], 'type': a.get('type'),
                       'evidence_span': a.get('evidence_span'),
                       'bot_sourced': bool(a.get('bot_sourced'))} for a in anchors],
          'event_id': cls.get('event_id'),
          'provenance': {'difficulty': cls['difficulty']['by'], 'op_type': cls['op_type']['by'],
                         'layer': cls['layer']['by'] if cls.get('layer') else 'rule:qtype',
                         'anchors': anchors[0].get('by') if anchors else None},
          'status': 'draft'}
    it.update(extra or {})
    return it


# These warnings are meant for human review, so the self-check also adds them to the
# arbitration queue (the reviewers' only input), not just to gates.jsonl.
# 这些 warn 需要人工处理, 因此同时写入仲裁队列。
GATE_TO_QUEUE = {'query_self_answer_contains': 'P5-4.9', 'query_self_answer_sim': 'P5-4.9',
                 'bot_sourced_mismatch': 'P5-botsrc'}


def self_gate(items, msgs, profile, d=None):
    by_turn = {m['turn']: m['text'] for m in msgs}
    bot_turns = frozenset(m['turn'] for m in msgs if m.get('kind') == 'bot')
    prof_by_turn = defaultdict(list)
    for p in profile:
        prof_by_turn[p.get('turn')].append(p)
    findings = []
    for it in items:
        fs = gates.q1_item(it) + gates.q2_item(it)
        fs += gates.q2_query_self_answer(it)  # query contains anchor text / high similarity
        fs += gates.q7_bot_only_memory(it, bot_turns)  # all anchors from the bot
        # Q7 applies to RT1 too: in proposal mode RT1 anchors are model-extracted as well, so they
        # need the literal check and the bot_sourced warning.
        anchors_by = (it.get('provenance') or {}).get('anchors') or ''
        if it['query_origin'] == 'authored' or anchors_by.startswith('model'):
            fs += gates.q7_item(it, by_turn, prof_by_turn, bot_turns)
        if it['query_origin'] == 'authored':  # Q8 applies to authored items only (native queries have no premise list)
            fs += gates.q8_item(it, by_turn) + gates.q8_posthoc(it)  # premises and hindsight wording
        fs += gates.q2_d3_leak(it)  # D3 self-answering
        errs = [f for f in fs if f['severity'] == 'error']
        if errs:
            it['status'] = 'flagged'
            # Feedback loop: gate failures go to the negative-example inbox for human annotation
            for f in errs:
                stage = 'rt2_query' if it['query_origin'] == 'authored' else 'query_specific'
                append_failure(stage, it['gid'], it['query'][:80], f['rule'], f['detail'])
        elif it['status'] != 'draft':
            it['status'] = 'built'
        if d:  # warnings also go to the human queue
            for f in fs:
                st = GATE_TO_QUEUE.get(f['rule'])
                if st:
                    append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                                 {'stage': st, 'iid': it['iid'], 'rule': f['rule'],
                                  'severity': f['severity'], 'detail': f['detail'],
                                  'reason': f"{f['rule']}(闸门粗筛, 判定权在人)"})
        findings += fs
    return findings


CONSUME_SIM = float(os.environ.get('CONSUME_SIM', '0.60'))


def consume_inventory(d, items):
    """Write each item's anchor use back to the inventory.
    Matching is two-level: the exact key (turn, fact[:12]), then fact similarity within the
    same turn (CONSUME_SIM), because an item may have reworded the fact. A miss on both is
    recorded as a warn in arbitration_queue instead of being skipped silently; lost usage
    would make RT2's unconsumed-first ordering pick the same material again.
    回写锚点使用记录: 先精确键匹配, 再按同轮次相似度匹配; 都不中则记 warn。
    """
    p = os.path.join(d, 'anchor_inventory.json')
    if not os.path.exists(p):
        return
    inv = json.load(open(p, encoding='utf-8'))
    by_key = {(a['turn'], (a['fact'] or '')[:12]): a for a in inv}
    by_turn = defaultdict(list)
    for a in inv:
        by_turn[a['turn']].append(a)
    miss = 0
    aq_p = os.path.join(d, 'arbitration_queue.jsonl')
    logged = {(r.get('iid'), r.get('turn')) for r in (load_jsonl(aq_p) if os.path.exists(aq_p) else [])
              if r.get('stage') == 'P5-consume'}  # record each miss only once; re-runs do not append again
    for it in items:
        for a in it['anchors']:
            tgt = by_key.get((a['turn'], (a['fact'] or '')[:12]))
            if tgt is None:
                f = _norm_fact(a.get('fact'))
                # key= compares ratios only: without it, max would compare the dict on ties and raise
                # TypeError. On ties within a turn the first candidate (inventory order) wins.
                cands = by_turn.get(a['turn'], [])
                b = max(cands, default=None,
                        key=lambda x: difflib.SequenceMatcher(None, f,
                                                              _norm_fact(x.get('fact'))).ratio())
                r = (difflib.SequenceMatcher(None, f, _norm_fact(b.get('fact'))).ratio()
                     if b is not None else 0.0)
                tgt = b if r >= CONSUME_SIM else None
            if tgt is None:
                miss += 1
                if (it['iid'], a['turn']) in logged:
                    continue
                logged.add((it['iid'], a['turn']))
                append_jsonl(aq_p,
                             {'stage': 'P5-consume', 'iid': it['iid'], 'turn': a['turn'],
                              'fact': (a.get('fact') or '')[:40],
                              'reason': f'消费记录写回失配(精确键与相似度回匹 <{CONSUME_SIM} 均未命中)'})
            elif it['iid'] not in tgt['consumed_by_items']:
                tgt['consumed_by_items'].append(it['iid'])
    json.dump(inv, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if miss:
        print(f'P5 消费回写: {miss} 条锚点未能回匹, 已入仲裁队列(stage=P5-consume)', flush=True)


def pair_items(items):
    """P5-6: link L1 <-> L2 items that share anchor turns (many-to-many)."""
    by_layer = {'L1': [], 'L2': []}
    for it in items:
        by_layer[it['layer']].append(it)
    for l2 in by_layer['L2']:
        t2 = {a['turn'] for a in l2['anchors']}
        l2['l1_pair'] = [l1['iid'] for l1 in by_layer['L1']
                         if t2 & {a['turn'] for a in l1['anchors']}]
    for l1 in by_layer['L1']:
        t1 = {a['turn'] for a in l1['anchors']}
        l1['l2_pair'] = [l2['iid'] for l2 in by_layer['L2']
                         if t1 & {a['turn'] for a in l2['anchors']}]


# ---------------- main ----------------

def _quota_arg(args, flag):
    """Quota argument as an upper bound. 'all' / 'auto' = driven by the material, no cap (stops
    when the pool is exhausted); a number is a hard upper bound. With a fixed target, the
    number of items would depend on run parameters rather than on the material.
    配额为上限; all/auto 表示由素材决定。
    """
    if flag not in args:
        return 0
    v = args[args.index(flag) + 1]
    return float('inf') if v in ('all', 'auto') else int(v)


def main():
    args = sys.argv[1:]
    gid = next((a for a in args if not a.startswith('-')), 'demo')
    do_rt1 = '--rt1' in args or not any(f in args for f in ('--rt2', '--rt2-d3', '--rt2-cf'))
    n_rt2 = _quota_arg(args, '--rt2')
    n_d3 = _quota_arg(args, '--rt2-d3')
    n_cf = _quota_arg(args, '--rt2-cf')
    no_llm = '--no-llm' in args
    strict_layer = '--strict-layer' in args  # skip positions whose layer is undecided
    limit = int(args[args.index('--limit') + 1]) if '--limit' in args else 0
    d = rundir(gid)
    msgs = load_msgs(d)
    who_by_turn = {m['turn']: m.get('who') for m in msgs}  # C6 needs the speaker of each anchor
    who_of = who_by_turn.get
    bot_turns = frozenset(m['turn'] for m in msgs if m.get('kind') == 'bot')  # bot turns, used to set bot_sourced
    profile = json.load(open(os.path.join(d, 'profile.json'), encoding='utf-8')) \
        if os.path.exists(os.path.join(d, 'profile.json')) else []
    track = next((m['src']['track'] for m in msgs
                  if isinstance(m.get('src'), dict) and m['src'].get('track')), 'jsonl')
    cands = {c['turn']: c for c in load_jsonl(os.path.join(d, 'candidates.jsonl'))}
    anchors_rows = {r['turn']: r for r in load_jsonl(os.path.join(d, 'anchors.jsonl'))}
    cls_rows = {r['turn']: r for r in load_jsonl(os.path.join(d, 'classify.jsonl'))}
    items_p = os.path.join(d, 'items.jsonl')
    items = load_jsonl(items_p)
    built_turns = {it.get('pos_turn') for it in items}  # resume key = service-point turn
    new_items = []

    if do_rt1:
        # No default pass: a position needs a P2 candidate row with route=RT1 and strong grounding.
        skip = {'no_candidate': 0, 'route': 0, 'grounding': 0}
        todo = []
        for t in sorted(anchors_rows):
            if t not in cls_rows or t in built_turns:
                continue
            c = cands.get(t)
            if c is None:
                skip['no_candidate'] += 1
            elif c.get('route') != 'RT1':
                skip['route'] += 1
            elif (c.get('grounding') or {}).get('v') != '强':
                skip['grounding'] += 1
            else:
                todo.append(t)
        if limit:
            todo = todo[:limit]
        print(f'P5-RT1: 可构位置 {len(todo)} · 跳过 {skip}', flush=True)
        for t in todo:
            ar, cl = anchors_rows[t], cls_rows[t]
            if not ar['anchors']:
                continue
            # Undecided layer: build the item as L1, mark it provisional, and send it to the
            # arbitration queue; do not skip it. L1 is the default because it needs no model call and
            # can be upgraded to L2 later by a re-run. --strict-layer skips such positions instead.
            # layer 未判定时按 L1 构题并标 provisional, 同时送仲裁队列; --strict-layer 改为跳过。
            layer = cl['layer']['v']
            layer_note = None
            if layer not in ('L1', 'L2'):
                if strict_layer:
                    append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                                 {'stage': 'P5-1', 'turn': t,
                                  'reason': 'layer_pending(--strict-layer 跳过)'})
                    continue
                # 'both' = a human judged the position valid as L1 or L2; built as L1 and noted, so an L2
                # counterpart can be added later.
                layer_note = 'both' if layer == 'both' else 'pending'
                layer = 'L1'
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P5-1', 'turn': t, 'layer_note': layer_note,
                              'reason': f'layer_{layer_note} → 按 L1 降级构题'
                                        f'(provisional, 需人工确认)'})
            cand = cands[t]
            iid = next_iid(items + new_items, gid, layer)
            conflict = cl['difficulty'].get('evidence')
            if layer == 'L1':
                atoms, distract = build_gt_atoms(ar['anchors'], conflict)
                extra = {'gt_atoms': atoms, 'distractor_atoms': distract,
                         'gt': '；'.join(atoms)}
            else:
                crits = grounding_criteria(ar['anchors'], gid, iid, cl['op_type']['v'])
                crits += penalty_criteria(ar['anchors'], profile, conflict,
                                          cand.get('who'), cand.get('src_turn'), iid, len(crits),
                                          grounded=grounded_turns(crits), who_of=who_of,
                                          rel_ctx=_bigrams(cand.get('query', '')) |
                                                  _bigrams(' '.join(c['check'] for c in crits)))
                extra = {'criteria': crits}
            it = make_item(gid, iid, layer, cand, ar['anchors'], cl, 'RT1', extra, track=track)
            it['pos_turn'] = t
            if layer_note:  # degraded output: marked explicitly, never built
                it['layer_provisional'] = True
                it['layer_note'] = layer_note
                it['provenance']['layer'] = f'rule:p5_default_L1({layer_note})'
            if layer == 'L2' and not no_llm:
                try:
                    it['criteria'] += qs_criteria(d, it, iid, len(it['criteria']))
                    it['criteria'] = enforce_caps(it['criteria'])  # must run after query_specific
                    it['status'] = 'built'
                except Exception as e:               # noqa: BLE001
                    # Any error on one item, not only ParseFail, sends it to the arbitration queue and the
                    # loop continues. New items are written only at the end of main(), so an uncaught error
                    # would lose the whole stage. Transient errors are already retried in llm.py; what
                    # reaches here is permanent.
                    # 单题出任何异常都送仲裁队列, 其余继续; 新题在 main() 末尾统一落盘, 未捕获的异常会丢掉整个阶段。
                    append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                                 {'stage': 'P5-5qs', 'iid': iid,
                                  'err': f'{type(e).__name__}: {str(e)[:200]}'})
            if layer == 'L2' and no_llm:
                it['criteria'] = enforce_caps(it['criteria'])  # --no-llm path
            elif layer == 'L1' and not layer_note:
                it['status'] = 'built'  # provisional items stay draft until a human confirms the layer
            new_items.append(it)

    if n_d3 and D3_V2:  # revised D3: conflict pair + same-matter aggregation (default)
        if no_llm:
            raise SystemExit('RT2-D3 题面反写需要模型, 不能与 --no-llm 同用')
        inv = json.load(open(os.path.join(d, 'anchor_inventory.json'), encoding='utf-8'))
        pairs = d3v2_dedupe_pairs(load_d3_pairs(d))
        if D3_PAIRS:  # build only the listed conflict pairs
            pairs = [pr for pr in pairs if f"{pr['old_anchor_turn']}->{pr['update_turn']}" in D3_PAIRS]
            print(f'P5-RT2D3(v2): D3_PAIRS 限定 {sorted(D3_PAIRS)} → 命中 {len(pairs)} 对', flush=True)
        pairs, blocked = d3v2_r7_filter(d, pairs)  # screening verdicts first
        done_keys, done_turns, done_sets = d3v2_done(d, items)  # reuse: pairs already built or with a terminal outcome are not redone
        is_done = lambda pr: (f"{pr['old_anchor_turn']}->{pr['update_turn']}" in done_keys
                              or pr['update_turn'] in done_turns
                              or any(pr['old_anchor_turn'] in s and pr['update_turn'] in s for s in done_sets))
        todo = [pr for pr in pairs if not is_done(pr)]
        if len(todo) < len(pairs):
            print(f'P5-RT2D3(v2): 复用 {len(pairs) - len(todo)} 对已处理的冲突对(已出题或已有取料结论), '
                  f'本轮处理 {len(todo)} 对', flush=True)
        pairs = todo
        blocked = [(pr, why) for pr, why in blocked if not is_done(pr)]
        for pr, why in blocked:
            _d3v2_log(d, pr,
                         {'stage': 'P5-1d3r7', 'update_turn': pr['update_turn'], 'old_anchor_turn': pr['old_anchor_turn'],
                          'reason': f'冲突对筛查未通过, 不出题: {why}'})
        cap = '素材驱动(不封顶)' if n_d3 == float('inf') else f'上限 {n_d3} 道'
        print(f'P5-RT2D3(v2): 冲突对 {len(pairs)} 组(去重+筛查后, 拦 {len(blocked)}), {cap}, '
              f'{"每对 L1+L2" if D3_V2_BOTH_LAYERS else "L1/L2 轮替"}', flush=True)
        built = 0
        for pi, pr in enumerate(pairs):
            if built >= n_d3:
                break
            form = D3_V2_FORMS[pi % len(D3_V2_FORMS)]  # service / trace alternate by pair
            core = d3_anchor_group(pr, inv, msgs)
            if not core:
                _d3v2_log(d, pr,
                             {'stage': 'P5-1d3', 'update_turn': pr.get('update_turn'),
                              'reason': 'D3 取料失败: 旧值锚或新值锚缺失'})
                continue
            old_a, new_a = core
            cut_turn = d3v2_cut(pr, msgs, pairs)
            if cut_turn is None:
                _d3v2_log(d, pr,
                             {'stage': 'P5-1d3span', 'update_turn': pr.get('update_turn'),
                              'reason': f'D3 跳过: 更新后不足 {RT2_MIN_SPAN} 轮或紧接同事项再次更新'})
                continue
            # Check that the new value is still current. If it changed again later, move the cutoff
            # before that change; if the span becomes too short, drop the pair rather than build an
            # item whose answer is already outdated.
            # 核查新值是否仍然有效; 之后又被改过就把截断点移到改动之前, 跨度不足则放弃。
            try:
                cur_ok, changed_turn, cur_note = d3v2_currency(d, pr, msgs, cut_turn)
            except Exception as e:                       # noqa: BLE001
                _d3v2_log(d, pr,
                             {'stage': 'P5-1d3v2', 'update_turn': pr['update_turn'],
                              'err': f'currency: {type(e).__name__}: {str(e)[:200]}'})
                continue
            if not cur_ok:
                moved = None
                if changed_turn and changed_turn - pr['update_turn'] > RT2_MIN_SPAN:
                    ts = [t for t in sorted(m['turn'] for m in msgs)
                          if pr['update_turn'] + RT2_MIN_SPAN <= t < changed_turn]
                    moved = ts[len(ts) // 2] if ts else None
                _d3v2_log(d, pr,
                             {'stage': 'P5-1d3stale', 'update_turn': pr['update_turn'], 'cut_turn': cut_turn,
                              'changed_turn': changed_turn, 'note': cur_note,
                              'reason': ('新值在截断点前已再次变更; 截断点前移至 #%d' % moved) if moved
                                        else '新值在截断点前已再次变更且无法前移截断点, 弃'})
                if moved is None:
                    continue
                cut_turn = moved
            cands = d3v2_candidates(pr, inv, cut_turn, {old_a['turn'], new_a['turn']})
            if not cands:
                _d3v2_log(d, pr,
                             {'stage': 'P5-1d3v2', 'update_turn': pr['update_turn'],
                              'reason': 'v2 取料失败: 窗口内无可聚合的实质锚点'})
                continue
            try:
                rel, why = d3v2_relate(d, pr, cands, who_of)
            except Exception as e:  # noqa: BLE001 (per-item error isolation, see main)
                _d3v2_log(d, pr,
                             {'stage': 'P5-1d3v2', 'update_turn': pr['update_turn'],
                              'err': f'{type(e).__name__}: {str(e)[:200]}'})
                continue
            related = d3v2_choose(rel, who_of(pr['update_turn']), who_of)
            if not related:
                _d3v2_log(d, pr,
                             {'stage': 'P5-1d3v2', 'update_turn': pr['update_turn'],
                              'reason': f'v2 取料失败: 同事项锚点 {len(rel)} 条(<{D3V2_N_RELATED[0]}) 或全部来自更新者本人',
                              'n_cands': len(cands)})
                continue
            group = sorted([old_a, new_a] + related, key=lambda a: a['turn'])
            anchors = anchor_views(group, bot_turns)
            conflict = [{'old_anchor_turn': pr['old_anchor_turn'], 'update_turn': pr['update_turn'],
                         'old_vals': [pr.get('old_value')], 'new_vals': [pr.get('new_value')]}]
            layers = ('L2', 'L1') if D3_V2_BOTH_LAYERS else (QTYPE_LAYER.get(D3_QTYPES[built % len(D3_QTYPES)], 'L1'),)
            for layer in layers:
                if built >= n_d3:
                    break
                qtype = d3v2_qtype(layer, form)
                try:
                    v = d3v2_query(d, msgs, anchors, related, layer, gid, form=form)
                except Exception as e:                   # noqa: BLE001
                    _d3v2_log(d, pr,
                                 {'stage': 'P5-1d3v2', 'update_turn': pr['update_turn'], 'layer': layer,
                                  'err': f'{type(e).__name__}: {str(e)[:200]}'})
                    continue
                iid = next_iid(items + new_items, gid, layer)
                cls = {'difficulty': {'v': 'D3', 'by': 'rule:d3v2(conflict_pair+aggregate)',
                                      'conflict_kind': 'update', 'evidence': conflict},
                       'op_type': {'v': 'resolve', 'by': 'rule:qtype_assign'},
                       'layer': {'by': 'rule:qtype'}, 'event_id': None}
                cand = {'query': v.get('query', ''), 'src_turn': None}
                if layer == 'L1' and form == 'trace':
                    # trace form: old values are legitimate answer atoms (links of the chain), and the
                    # "old value is void" atom is reachable because the query asks for the chain
                    atoms = [f"{a['fact']}(#{a['turn']})" for a in anchors]
                    atoms += [a for a in build_gt_atoms(anchors, conflict)[0] if a.startswith('此前的「')]
                    extra = {'gt_atoms': atoms, 'distractor_atoms': [], 'gt': '；'.join(atoms)}
                elif layer == 'L1':
                    atoms, distract = build_gt_atoms(anchors, conflict, stale_decl=False)
                    decl = [a for a in build_gt_atoms(anchors, conflict)[0] if a.startswith('此前的「')]
                    extra = {'gt_atoms': atoms, 'distractor_atoms': distract, 'gt': '；'.join(atoms),
                             'gt_stale_decl': decl}
                else:
                    # service: the old-value anchor is not grounding; trace: it is one link of the chain and
                    # is legitimate grounding
                    live = anchors if form == 'trace' else [a for a in anchors if a['turn'] != old_a['turn']]
                    _, distract = build_gt_atoms(anchors, conflict)
                    crits = grounding_criteria(live, gid, iid, 'aggregate')  # folded, same structure as aggregate D2 items
                    c5_check = (f"把已被覆盖的旧值「{old_a['fact']}」(#{old_a['turn']})当作最终结论, 未以 #{new_a['turn']} 之后生效的值收尾"
                                if form == 'trace' else
                                f"按 #{old_a['turn']} 的旧值「{old_a['fact']}」回答, 未采用 #{new_a['turn']} 更新后的值")
                    c5 = {'cid': f'{iid}-c{len(crits) + 1}', 'check': c5_check,
                          'kind': 'penalty', 'polarity': None, 'points': -2,
                          'source': 'feedback(C5)', 'by': 'rule:cluster_trigger'}
                    pen = penalty_criteria(anchors, profile, None, None, cut_turn, iid, len(crits) + 1,
                                           grounded=grounded_turns(crits), who_of=who_of,
                                           rel_ctx=_bigrams(cand['query']) |
                                                   _bigrams(' '.join(c['check'] for c in crits)))
                    crits = crits + [c5] + pen[:2]
                    extra = {'criteria': crits, 'distractor_atoms': distract}
                extra.update(premises=v.get('premises') or [], qtype=qtype, cut_turn=cut_turn,
                             conflict_kind='update',
                             d3v2={'pair_key': f"{pr['old_anchor_turn']}->{pr['update_turn']}",
                                   'form': form, 'related_turns': [a['turn'] for a in related], 'relate_why': why,
                                   'n_cands': len(cands), 'n_related_raw': len(rel),
                                   'currency': cur_note, 'cut_moved_from': None if cur_ok else changed_turn})
                it = make_item(gid, iid, layer, cand, anchors, cls, 'RT2', extra, track=track)
                if layer == 'L2':
                    try:
                        it['criteria'] += qs_criteria(d, it, iid, len(it['criteria']))
                    except Exception as e:               # noqa: BLE001
                        _d3v2_log(d, pr,
                                     {'stage': 'P5-5qs', 'iid': iid,
                                      'err': f'{type(e).__name__}: {str(e)[:200]}'})
                    it['criteria'] = enforce_caps(it['criteria'])
                it['status'] = 'built'
                new_items.append(it)
                built += 1
                print(f"  {iid} [D3v2-{form}/{layer}] 旧#{pr['old_anchor_turn']} → 新#{pr['update_turn']} "
                      f"+聚合{[a['turn'] for a in related]} cut={cut_turn} {v.get('query', '')[:40]}", flush=True)

    elif n_d3:  # legacy D3 path (D3_V2=0)
        if no_llm:
            raise SystemExit('RT2-D3 题面反写需要模型, 不能与 --no-llm 同用')
        inv = json.load(open(os.path.join(d, 'anchor_inventory.json'), encoding='utf-8'))
        pairs = load_d3_pairs(d)
        cap = '素材驱动(不封顶)' if n_d3 == float('inf') else f'上限 {n_d3} 道'
        print(f'P5-RT2D3: 冲突对 {len(pairs)} 组(来自 p4_d3), {cap}', flush=True)
        built = 0
        for pr in pairs:
            if built >= n_d3:
                break
            group = d3_anchor_group(pr, inv, msgs)
            if not group:
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P5-1d3', 'update_turn': pr.get('update_turn'),
                              'reason': 'D3 取料失败: 旧值锚或新值锚缺失'})
                continue
            qtype = D3_QTYPES[built % len(D3_QTYPES)]  # alternate L1 / L2
            layer = QTYPE_LAYER.get(qtype, 'L1')
            # The D3 cutoff must come after the update; otherwise the new value does not exist yet
            # when the question is asked.
            cut_turn = rt2_cut_turn(pr['update_turn'], msgs)
            if cut_turn is None:
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P5-1d3span', 'update_turn': pr.get('update_turn'),
                              'reason': f'D3 跳过: 更新后不足 {RT2_MIN_SPAN} 轮'})
                continue
            try:
                v = rt2_query(d, msgs, profile, group, qtype, gid=gid,
                              q_budget=D3_QUERY_BUDGET.get(layer))
            except Exception as e:  # noqa: BLE001 (per-item error isolation, see main)
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P5-1d3', 'update_turn': pr['update_turn'],
                              'err': f'{type(e).__name__}: {str(e)[:200]}'})
                continue
            iid = next_iid(items + new_items, gid, layer)
            anchors = anchor_views(group, bot_turns)  # bot_sourced from the transcript
            conflict = [{'old_anchor_turn': pr['old_anchor_turn'],
                         'update_turn': pr['update_turn'],
                         'old_vals': [pr.get('old_value')], 'new_vals': [pr.get('new_value')]}]
            cls = {'difficulty': {'v': 'D3', 'by': 'rule:d3_pair', 'conflict_kind': 'update',
                                  'evidence': conflict},
                   'op_type': {'v': 'resolve', 'by': 'rule:qtype_assign'},
                   'layer': {'by': 'rule:qtype'}, 'event_id': None}
            cand = {'query': v.get('query', ''), 'src_turn': None}
            if layer == 'L1':
                atoms, distract = build_gt_atoms(anchors, conflict)  # old-value anchor -> distractor
                extra = {'gt_atoms': atoms, 'distractor_atoms': distract, 'gt': '；'.join(atoms)}
            else:
                # L2-D3 grounding uses only the new-value anchors. Grounding on the superseded value would
                # require the answer to use it, which collides with C5 (penalty for answering with the old
                # value). The old value is recorded in distractor_atoms and penalised by C5, as in
                # build_gt_atoms for L1.
                # L2-D3 的 grounding 只取新值锚; 旧值记入干扰项并由 C5 扣分。
                old_turns = {h['old_anchor_turn'] for h in conflict}
                fresh = [a for a in anchors if a['turn'] not in old_turns] or anchors
                _, distract = build_gt_atoms(anchors, conflict)
                crits = grounding_criteria(fresh, gid, iid, 'resolve')
                crits += penalty_criteria(fresh, profile, conflict, None, cut_turn, iid,
                                          len(crits), grounded=grounded_turns(crits),
                                          who_of=who_of,
                                          rel_ctx=_bigrams(v.get('query', '')) |
                                                  _bigrams(' '.join(c['check'] for c in crits)))
                extra = {'criteria': crits, 'distractor_atoms': distract}
            extra.update(premises=v.get('premises') or [], qtype=qtype, cut_turn=cut_turn,
                         conflict_kind='update')
            it = make_item(gid, iid, layer, cand, anchors, cls, 'RT2', extra, track=track)
            if layer == 'L2':
                try:
                    it['criteria'] += qs_criteria(d, it, iid, len(it['criteria']))
                except Exception as e:  # noqa: BLE001 (per-item error isolation, see main)
                    append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                                 {'stage': 'P5-5qs', 'iid': iid,
                                  'err': f'{type(e).__name__}: {str(e)[:200]}'})
                it['criteria'] = enforce_caps(it['criteria'])
            it['status'] = 'built'
            new_items.append(it)
            built += 1
            print(f"  {iid} [D3/{layer}] 旧#{pr['old_anchor_turn']} → 新#{pr['update_turn']} "
                  f"cut={cut_turn} {v.get('query', '')[:44]}", flush=True)

    if n_rt2:
        if no_llm:
            raise SystemExit('RT2 题面反写需要模型, 不能与 --no-llm 同用')
        # Reuse: RT2 is material-driven, so re-running it on a finished run would keep adding
        # new authored items on top of the existing ones. New items are only written at the
        # end of this function, so existing RT2 items mean the stage completed. Pass
        # --rt2-append to add more anyway.
        # 复用: RT2 按素材出题, 在已完成的 run 上重跑会在原题之上继续追加。新题只在函数末尾
        # 统一落盘, 故已有 RT2 题 = 该阶段已完成。确需追加用 --rt2-append。
        prior_rt2 = [it for it in items if it.get('query_origin') == 'authored' and not it.get('d3v2')
                     and it.get('conflict_kind') not in ('update', 'arbitration')]
        if prior_rt2 and '--rt2-append' not in args:
            print(f'P5-RT2: 复用已有 {len(prior_rt2)} 道 RT2 题, 不再出新题; 追加请加 --rt2-append', flush=True)
            n_rt2 = 0
    if n_rt2:
        inv = json.load(open(os.path.join(d, 'anchor_inventory.json'), encoding='utf-8'))
        # RT2 does not use bot_sourced anchors as material (bot-turn facts not confirmed by a human)
        rt2_used = defaultdict(int)  # anchor use counts in this RT2 pass
        for it in items:
            if it.get('query_origin') == 'authored':
                for a in it.get('anchors') or []:
                    rt2_used[a['turn']] += 1
        seen_groups = [{a['turn'] for a in it.get('anchors') or []}
                       for it in items if it.get('query_origin') == 'authored']
        pool = rt2_pool(inv, rt2_used)
        cap = '素材驱动(不封顶)' if n_rt2 == float('inf') else f'上限 {n_rt2} 道'
        print(f'P5-RT2: 选料池 {len(pool)}, {cap}', flush=True)
        qi = 0
        built_rt2 = 0
        for a in pool:
            if built_rt2 >= n_rt2:
                break
            qtype = QTYPES[qi % len(QTYPES)]
            qi += 1
            layer = QTYPE_LAYER.get(qtype, 'L1')
            if rt2_used[a['turn']] >= RT2_ANCHOR_MAX_USE:
                continue
            group = rt2_pick_group(a, pool, qtype)  # anchor group
            if not rt2_group_ok(group, seen_groups):  # near-duplicate group
                continue
            # Decide the cutoff first. If the span is too short (anchors near the end of the timeline)
            # or the anchor is updated within the interval, skip: no items whose answer is in the
            # previous message or already outdated.
            # 先定截断点; 跨度不足或锚点在区间内被更新则跳过。
            cut_turn = rt2_cut_turn(max(x['turn'] for x in group), msgs,
                                    detect_stale=lambda at, ct: bool(
                                        p4_detect_update([{'turn': at, 'fact': a['fact'],
                                                           'type': a.get('type')}], msgs, ct)))
            if cut_turn is None:
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P5-1rt2span', 'anchor_turn': a['turn'],
                              'reason': f'RT2 跳过: 锚点后不足 {RT2_MIN_SPAN} 轮或区间内已被更新'})
                continue
            try:
                v = rt2_query(d, msgs, profile, group, qtype, gid=gid)
            except Exception as e:  # noqa: BLE001 (per-item error isolation, see main)
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P5-1rt2', 'anchor_turn': a['turn'],
                              'err': f'{type(e).__name__}: {str(e)[:200]}'})
                continue
            iid = next_iid(items + new_items, gid, layer)
            anchors = anchor_views(group, bot_turns)  # bot_sourced from the transcript
            # difficulty follows the same rule as RT1, never hard-coded
            n_units = len(p4_substantive_units(anchors))
            cls = {'difficulty': {'v': 'D2' if n_units >= P4_D2_MIN else 'D1',
                                  'by': 'rule:memory_units', 'n_units': n_units},
                   'op_type': {'v': QTYPE_OP[qtype], 'by': 'rule:qtype_assign'},
                   'layer': {'by': 'rule:qtype'}, 'event_id': None}
            cand = {'query': v.get('query', ''), 'src_turn': None}
            if layer == 'L1':
                atoms, distract = build_gt_atoms(anchors, None)
                extra = {'gt_atoms': atoms, 'distractor_atoms': distract, 'gt': '；'.join(atoms)}
            else:
                crits = grounding_criteria(anchors, gid, iid, QTYPE_OP[qtype])
                # Pass the real cutoff so penalty_criteria filters profile entries by turn; otherwise
                # future profile entries could enter the criteria.
                # 传入真实截断点, 让扣分项按轮次过滤档案条目, 防止引用未来的档案。
                crits += penalty_criteria(anchors, profile, None, None, cut_turn, iid, len(crits),
                                          grounded=grounded_turns(crits), who_of=who_of,
                                          rel_ctx=_bigrams(cand.get('query', '')) |
                                                  _bigrams(' '.join(c['check'] for c in crits)))
                extra = {'criteria': crits}
            extra['premises'] = v.get('premises') or []
            extra['qtype'] = qtype
            extra['cut_turn'] = cut_turn
            for x in group:  # record reuse counts and the group
                rt2_used[x['turn']] += 1
            seen_groups.append({x['turn'] for x in group})
            it = make_item(gid, iid, layer, cand, anchors, cls, 'RT2', extra, track=track)
            if layer == 'L2':
                try:
                    it['criteria'] += qs_criteria(d, it, iid, len(it['criteria']))
                except Exception as e:  # noqa: BLE001 (per-item error isolation, see main)
                    append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                                 {'stage': 'P5-5qs', 'iid': iid,
                                  'err': f'{type(e).__name__}: {str(e)[:200]}'})
                it['criteria'] = enforce_caps(it['criteria'])  # count limits
            built_rt2 += 1
            it['status'] = 'built'
            new_items.append(it)
            print(f"  {iid} [{qtype}] {it['query'][:40]}", flush=True)

    if n_cf:  # arbitration-type D3
        # Kept as a separate block from the rt2-d3 loop. Material = resolved conflicts from
        # p4_conflict (already validated in code: spans, speakers, distinct members, order); this
        # block only assembles and builds. Errors are caught as Exception, not only ParseFail, as
        # elsewhere. No query length budget: gates.q2_d3_leak applies only to conflict_kind=update.
        # 独立于 rt2-d3 循环。素材为 p4_conflict 中已消解且已通过代码校验的冲突; 本块只负责组装与构题。
        if no_llm:
            raise SystemExit('RT2-CF 题面反写需要模型, 不能与 --no-llm 同用')
        cf_pairs = [r for r in load_jsonl(os.path.join(d, 'd3_conflicts.jsonl'))
                    if r.get('resolved') and (r.get('resolution') or {}).get('turn') is not None]
        cap = '素材驱动(不封顶)' if n_cf == float('inf') else f'上限 {n_cf} 道'
        print(f'P5-RT2CF: 已消解冲突对 {len(cf_pairs)} 组(来自 p4_conflict), {cap}', flush=True)
        done_cf = {it.get('cf_turn') for it in items if it.get('conflict_kind') == 'arbitration'}
        built_cf = 0
        for k, pr in enumerate(cf_pairs):
            if built_cf >= n_cf:
                break
            res = pr['resolution']
            if res['turn'] in done_cf:
                continue
            qtype = '多人冲突裁决' if k % 2 else '冲突裁决结果'  # alternate L2 / L1
            layer = QTYPE_LAYER.get(qtype, 'L1')
            anchors = ([{'turn': p['turn'], 'fact': f"{p['member']}主张{p['stance']}",
                         'type': '事实', 'by': pr.get('by'), 'evidence_span': p.get('span')}
                        for p in pr.get('parties') or []]
                       + [{'turn': res['turn'],
                           'fact': f"关于{pr.get('topic')}的最终结果为{res.get('outcome')}",
                           'type': '事实', 'by': pr.get('by'), 'evidence_span': res.get('span')}])
            cut_turn = rt2_cut_turn(res['turn'], msgs)
            if cut_turn is None:
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P5-1cfspan', 'cf_turn': res['turn'],
                              'reason': f'RT2-CF 跳过: 裁决后不足 {RT2_MIN_SPAN} 轮'})
                continue
            try:
                v = rt2_query(d, msgs, profile, anchors, qtype, gid=gid)
            except Exception as e:  # noqa: BLE001 (per-item error isolation, see main)
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P5-1cf', 'cf_turn': res['turn'],
                              'err': f'{type(e).__name__}: {str(e)[:200]}'})
                continue
            iid = next_iid(items + new_items, gid, layer)
            cls = {'difficulty': {'v': 'D3', 'by': 'rule:cf_pair', 'conflict_kind': 'arbitration',
                                  'evidence': [{'party_turns': [p['turn'] for p in pr['parties']],
                                                'resolution_turn': res['turn']}]},
                   'op_type': {'v': 'resolve', 'by': 'rule:qtype_assign'},
                   'layer': {'by': 'rule:qtype'}, 'event_id': None}
            cand = {'query': v.get('query', ''), 'src_turn': None}
            if layer == 'L1':
                # Answer = the outcome + both positions (all with #turn). No distractor: who "lost" cannot
                # be decided mechanically (in a compromise both give way), and the losing position is
                # history, not a wrong answer.
                # 答案 = 裁决结果 + 双方立场; 不设干扰项, 败方立场是事实而非错误答案。
                atoms = [f"{a['fact']}(#{a['turn']})" for a in anchors]
                extra = {'gt_atoms': atoms, 'distractor_atoms': [], 'gt': '；'.join(atoms)}
            else:
                crits = grounding_criteria(anchors, gid, iid, 'resolve')
                crits.append({'cid': f'{iid}-c{len(crits) + 1}',
                              'check': f"忽略 #{res['turn']} 已达成的裁决结果, 或按被搁置的"
                                       f"单方主张给出安排",
                              'kind': 'penalty', 'polarity': None, 'points': -2,
                              'source': f"cf#{res['turn']}", 'by': 'rule:cf_penalty'})
            if layer == 'L2':
                extra = {'criteria': crits}
            extra.update(premises=v.get('premises') or [], qtype=qtype, cut_turn=cut_turn,
                         conflict_kind='arbitration')
            it = make_item(gid, iid, layer, cand, anchors, cls, 'RT2', extra, track=track)
            it['cf_turn'] = res['turn']
            if layer == 'L2':
                try:
                    it['criteria'] += qs_criteria(d, it, iid, len(it['criteria']))
                except ParseFail:
                    pass
                it['criteria'] = enforce_caps(it['criteria'])
            it['status'] = 'built'
            new_items.append(it)
            built_cf += 1
            print(f"  {iid} [D3-arb/{layer}] {pr.get('topic')} 裁决#{res['turn']} "
                  f"cut={cut_turn} {v.get('query', '')[:40]}", flush=True)

    findings = self_gate(new_items, msgs, profile, d=d)
    pair_items(items + new_items)
    for it in new_items:
        append_jsonl(items_p, it)
    # Corpus-level gates on the FULL item set, written to gates.jsonl. Template share and atom
    # citation coverage are corpus-level measures, so computing them on the new items of one
    # pass would be meaningless; the last of the three passes has the complete numbers.
    # 语料级闸门按全量题目运行并落盘。
    findings += gates.corpus_checks(items + new_items)
    append = os.path.join(d, 'gates.jsonl')
    if new_items:
        for f in findings:
            append_jsonl(append, f)
    else:  # reuse: no new items, so the corpus-level result is unchanged and not appended again
        print('P5: 本轮无新题, 闸门记录沿用已有 gates.jsonl', flush=True)
    # Rebuild consumption from the FULL item set: on a resumed run new_items is nearly empty,
    # and a P3 re-run rebuilds the inventory and clears `consumed`.
    # 按全量题目重建锚点使用记录。
    consume_inventory(d, items + new_items)
    from collections import Counter
    st = Counter(it['status'] for it in new_items)
    sev = Counter(f['severity'] for f in findings)
    print(f'P5 汇总: 新题 {len(new_items)} {dict(st)} · 闸门 findings {dict(sev)}')
    print(f'→ {items_p}')


if __name__ == '__main__':
    main()
