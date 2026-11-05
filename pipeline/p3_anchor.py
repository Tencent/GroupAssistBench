#!/usr/bin/env python3
# P3 Anchor memories (paper 3.2 "anchoring memories"): for each retained request,
# extract the minimal set of facts the answer depends on. Every anchor carries a
# verbatim evidence span that is verified in code against its source.
#   P3-1 context (code)          reuses p2_locate.build_ctx (CTX_FULL controls full history)
#   P3-2 extraction (model)      candidates + counterfactual self-check in one call; the
#                                evidence span is verified against the message, or against
#                                a profile entry when the anchor cites the member profile
#   P3-3 earliest mention (code) values and key phrases are searched across the whole
#                                history; an earlier hit raises earlier_hit
#   P3-4 fact check + audit (code) pointer-like anchors are rejected; entities the request
#                                mentions but no anchor covers are recorded for audit
# Outputs: anchors.jsonl (per position) + anchor_inventory.json (with `consumed` flags)
# Usage: python3 pipeline/p3_anchor.py <gid> [--from-gold] [--limit N] [--turns a,b] [--workers N]
#   --from-gold  load the human anchors from annotations_gold directly (no API); the
#                P3-3 / P3-4 checks still run, since human anchors can also miss the
#                earliest mention
# P3 锚定: 找出每条请求所依赖的历史事实; 每个锚点附原文片段并在代码中校验。
import difflib, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.llm import (config_banner, MODEL_MAIN, append_jsonl, append_if_absent,
                         dedup_jsonl_by, chat_json, load_jsonl, span_in, workers_arg, ParseFail)
from pipeline.p2_locate import rundir, load_msgs, build_ctx
from pipeline.examples import render as render_examples

POINTER_PAT = re.compile(r'依\s*#\d+|对比判断|最终确定的方案|统计各户表态|见上|如前所述|参考上文')
NUM_PAT = re.compile(r'\d+(?:\.\d+)?')
ANCHOR_TYPES = ('事实', '承诺', '约束', '偏好', '人设')

ANC_SYS = """你是数据标注流水线中的「anchor 抽取」模块。给定群成员档案、全群历史与一条提问，找出回答该提问必须依赖的历史事实(锚点)。输出会被程序校验: evidence_span 必须是所指 turn 消息原文(或成员档案条目)的连续子串, 校验失败该条即被丢弃。

# 两步执行
① 候选: 列出提问所依赖的每个历史事实, 给出它**首次确立**的 turn(不是最近提及)与原文片段
② 自筛: 对每条候选做反事实——拿掉它, 回答会变差/变错吗? 不会则删

# 硬规则
- fact 写具体值不写指针(✗「他提过的计划」✓「品胜随身WiFi已下单未到货」)
- 成员的长期特质(习惯/身份/人设)优先引用成员档案条目, turn 用档案给出的首次出现 turn
- 宁缺毋滥; 与提问无关的背景不录
- 反例警示: 曾有条目把「无障碍按钮讨论」错挂为「低层费用」的锚点——turn 指错整条作废

# 输出严格 JSON
{"anchors":[{"turn":123,"fact":"具体值","type":"事实|承诺|约束|偏好|人设","evidence_span":"该turn原文连续子串(≤25字)"}]}"""


def verify_anchor(e, by_turn, prof_by_turn):
    """Verify evidence_span against its source (normalised match, see llm.span_in):
    message text -> 'msg'; profile quote/fact -> 'profile'.
    校验 evidence_span: 命中消息原文记 msg, 命中档案条目记 profile。
    """
    t, span = e.get('turn'), e.get('evidence_span') or ''
    if not isinstance(t, int) or not span:
        return None
    if span_in(span, by_turn.get(t)):
        return 'msg'
    for p in prof_by_turn.get(t, []):
        if span_in(span, p.get('quote')) or span_in(span, p.get('fact')):
            return 'profile'
    return None


def earliest_backscan(e, msgs):
    """P3-3 (code): search the whole history for the values in `fact` and the head of the
    evidence span. An earlier hit only raises a warning; the anchor is not rewritten.
    在全部历史中回扫 fact 的数值与片段; 更早出现只报警, 不改写锚点。
    """
    keys = set(NUM_PAT.findall(e.get('fact') or ''))
    span = (e.get('evidence_span') or '')[:12]
    if span:
        keys.add(span)
    keys = {k for k in keys if len(k) >= 2}
    if not keys:
        return None
    hits = [m['turn'] for m in msgs
            if m['turn'] < e['turn'] and any(k in (m['text'] or '') for k in keys)]
    return min(hits) if hits else None


def audit_missing(query, anchors, profile):
    """P3-4 audit (code): member names or values in the request that no anchor covers.
    漏锚审计: 请求中提到、但没有任何锚点覆盖的成员名/数值。
    """
    covered = ''.join((a.get('fact') or '') + (a.get('evidence_span') or '') for a in anchors)
    missing = []
    for m in {p['member'] for p in profile}:
        name = m.split('（')[0]
        if name and name in query and name not in covered:
            missing.append(f'member:{name}')
    for n in NUM_PAT.findall(query):
        if len(n) >= 2 and n not in covered:
            missing.append(f'num:{n}')
    return missing


def process_position(d, msgs, profile, by_turn, prof_by_turn, turn, query, raw_anchors, by,
                     bot_turns=frozenset(), src_turn=None):
    ok, rejected = [], []
    is_model = by.startswith('model:')
    for e in raw_anchors:
        e = dict(e)
        e['by'] = by
        if not isinstance(e.get('turn'), int) or e['turn'] >= turn:
            e['reject'] = f'not_before_query(turn={e.get("turn")})'
            rejected.append(e)
            continue
        # Bot turns hold two kinds of content: generated knowledge answers, which are not group
        # memory, and restatements of group facts, which are. A mechanical rule cannot tell them
        # apart, so such anchors are kept but marked bot_sourced: RT2 does not use them as
        # material and the gates flag them for human confirmation.
        # bot 消息里既有通用知识回答(不算群记忆), 也有对群内事实的转述(算)。代码分不开,
        # 故保留但标记 bot_sourced: RT2 不取作素材, 闸门提示人工确认。
        if is_model and e['turn'] in bot_turns:
            e['bot_sourced'] = True
        if is_model and src_turn is not None and e['turn'] == src_turn:
            e['reject'] = 'anchor_is_query_itself'  # an anchor that is the request itself carries no information
            rejected.append(e)
            continue
        # Check the anchor type against ANCHOR_TYPES. An unknown type does not make the fact
        # wrong, so it is flagged and counted rather than rejected.
        # 校验锚点类型; 类型越界不代表事实错, 只标记计数, 不拒绝。
        if e.get('type') not in ANCHOR_TYPES:
            e['type_out_of_domain'] = e.get('type')
        if POINTER_PAT.search(e.get('fact') or ''):
            e['reject'] = 'pointer_not_fact'  # P3-4: a pointer is not a fact
            rejected.append(e)
            continue
        if by.startswith('model:'):
            v = verify_anchor(e, by_turn, prof_by_turn)
            if not v:
                e['reject'] = 'evidence_span_not_found'
                rejected.append(e)
                continue
            e['verified'] = v
        else:
            e['verified'] = 'human'  # human anchors have no evidence_span, so no literal check
        eh = earliest_backscan(e, msgs)
        if eh is not None and eh < e['turn']:
            e['earlier_hit'] = eh  # P3-3 warning: may not be the earliest mention
        ok.append(e)
    return {'turn': turn, 'query': query, 'anchors': ok, 'anchors_rejected': rejected,
            'audit_missing': audit_missing(query, ok, profile), 'by': by, 'status': 'proposal'}


INV_SIM = float(os.environ.get('INV_SIM', '0.75'))  # similarity threshold for merging anchors within a turn


def _inv_norm(s):
    """Normalise: strip punctuation, brackets, quotes and whitespace; keep content characters.
    归一化: 去掉标点、书名号、引号与空白, 只保留实义字符。
    """
    return re.sub(r'[《》「」“”"\'（）()、，。！？；：\s]', '', s or '')


def build_inventory(rows, inv_path):
    """Anchor inventory: anchors within the same turn are merged by normalised similarity,
    and each entry records which request positions use it; `consumed` is set by P5.
    Near-identical restatements of one fact would otherwise inflate the inventory and
    distort the pool of unconsumed anchors that RT2 draws from.
    When merging, the longest wording is kept and used_by_queries are combined.
    锚点库存: 同一轮次内按归一化相似度合并近义锚点, 保留最长表述并合并引用记录; consumed 由 P5 回写。
    """
    old = {}
    if os.path.exists(inv_path):
        for a in json.load(open(inv_path, encoding='utf-8')):
            old[(a['turn'], _inv_norm(a.get('fact')))] = a
    per_turn = defaultdict(list)  # turn -> [entry]
    for r in rows:
        for a in r['anchors']:
            t, fact = a['turn'], (a.get('fact') or '')
            nf = _inv_norm(fact)
            hit = None
            for e in per_turn[t]:
                if difflib.SequenceMatcher(None, nf, _inv_norm(e['fact'])).ratio() > INV_SIM:
                    hit = e
                    break
            if hit is None:
                prev = old.get((t, nf), {})
                hit = {'turn': t, 'fact': fact, 'type': a.get('type'), 'by': a.get('by'),
                       'evidence_span': a.get('evidence_span'),
                       'bot_sourced': a.get('bot_sourced', False),
                       'used_by_queries': [],
                       'consumed_by_items': prev.get('consumed_by_items', []),
                       'merged_variants': 0}
                per_turn[t].append(hit)
            else:
                hit['merged_variants'] += 1
                if len(fact) > len(hit['fact'] or ''):  # keep the most informative wording
                    hit['fact'], hit['evidence_span'] = fact, a.get('evidence_span')
                hit['bot_sourced'] = hit['bot_sourced'] or a.get('bot_sourced', False)
                if not hit['consumed_by_items']:
                    hit['consumed_by_items'] = old.get((t, nf), {}).get('consumed_by_items', [])
            hit['used_by_queries'].append(r['turn'])
    # Anchors that p4_d3 back-filled from the transcript are not in anchors.jsonl, so a
    # rebuild from rows alone would drop them. p4_d3 reuses its finished output and does
    # not back-fill again, so carry them over or the D3 pairs that need them lose material.
    # p4_d3 回填的锚点不在 anchors.jsonl 里, 只按 rows 重建会丢; 而 p4_d3 复用已有产物、
    # 不会再补, 故原样带过来, 否则依赖它们的 D3 冲突对会取料失败。
    for (t, nf), a in old.items():
        if str(a.get('by', '')).startswith('rule:d3_backfill') and not any(
                difflib.SequenceMatcher(None, nf, _inv_norm(e['fact'])).ratio() > INV_SIM
                for e in per_turn[t]):
            per_turn[t].append(a)
    out = sorted((e for v in per_turn.values() for e in v), key=lambda x: (x['turn'], x['fact']))
    for e in out:
        e['used_by_queries'] = sorted(set(e['used_by_queries']))
    json.dump(out, open(inv_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return out


def main():
    args = sys.argv[1:]
    gid = next((a for a in args if not a.startswith('-')), 'demo')
    from_gold = '--from-gold' in args
    limit = int(args[args.index('--limit') + 1]) if '--limit' in args else 0
    turns = ({int(x) for x in args[args.index('--turns') + 1].split(',')}
             if '--turns' in args else None)
    print(config_banner(), flush=True)
    d = rundir(gid)
    msgs = load_msgs(d)
    by_turn = {m['turn']: m['text'] for m in msgs}
    bot_turns = frozenset(m['turn'] for m in msgs if m.get('kind') == 'bot')
    profile = json.load(open(os.path.join(d, 'profile.json'), encoding='utf-8')) \
        if os.path.exists(os.path.join(d, 'profile.json')) else []
    prof_by_turn = {}
    for p in profile:
        prof_by_turn.setdefault(p.get('turn'), []).append(p)

    res_p = os.path.join(d, 'anchors.jsonl')
    done = {r['turn'] for r in load_jsonl(res_p)}

    if from_gold:
        gold = load_jsonl(os.path.join(d, 'annotations_gold.jsonl'))
        todo = [g for g in gold if g['anchors'] and g['turn'] not in done]
        if limit:
            todo = todo[:limit]
        print(f'P3(from-gold): 带锚点位置 {sum(1 for g in gold if g["anchors"])} · 本轮 {len(todo)}', flush=True)
        for g in todo:
            row = process_position(d, msgs, profile, by_turn, prof_by_turn,
                                   g['turn'], g['query'], g['anchors'],
                                   by=g['anchors'][0].get('by', 'human:?'))
            if not append_if_absent(res_p, row):
                print(f"  ⚠ #{g['turn']} 已存在, 本次结果丢弃(疑似跨进程重复跑)", flush=True)
    else:
        cands = [c for c in load_jsonl(os.path.join(d, 'candidates.jsonl'))
                 if c['route'] == 'RT1' and c['grounding']['v'] in ('强', '弱')]
        todo = [c for c in cands if c['turn'] not in done]
        if turns is not None:
            todo = [c for c in todo if c['turn'] in turns]
        if limit:
            todo = todo[:limit]
        print(f'P3(模型抽取 {MODEL_MAIN}): 候选 {len(cands)} · 本轮 {len(todo)}', flush=True)
        anc_sys = ANC_SYS + render_examples('anchor', exclude_gid=gid)  # rule 2: examples come from other groups

        def process(c):
            ctx = build_ctx(msgs, c['turn'], profile)
            try:
                v = chat_json(d, 'p3_anchor', anc_sys,
                              ctx + f"\n提问(#{c['turn']}): {c['query']}", max_tokens=1200)
            except ParseFail as e:
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P3-2', 'turn': c['turn'], 'err': str(e)})
                return
            row = process_position(d, msgs, profile, by_turn, prof_by_turn,
                                   c['turn'], c['query'], v.get('anchors') or [],
                                   by=f'model:main({MODEL_MAIN})',
                                   bot_turns=bot_turns, src_turn=c.get('src_turn'))
            wrote = append_if_absent(res_p, row)
            # A skipped turn means another process already wrote it; this result is discarded,
            # but the fact is printed rather than hidden.
            # 跳过意味着其他进程已写过该轮次; 本次结果丢弃, 但打印出来。
            print(f"  #{c['turn']:4d} anchors={len(row['anchors'])} "
                  f"rejected={len(row['anchors_rejected'])}"
                  f"{'' if wrote else '  ⚠ 已存在, 本次结果丢弃(疑似跨进程重复跑)'}", flush=True)

        def safe(c):
            """Per-candidate error isolation: any exception is recorded in the arbitration queue
            and the batch continues.
            候选级错误隔离: 任何异常都记入仲裁队列, 批次继续。
            """
            try:
                return process(c)
            except Exception as e:
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P3', 'turn': c['turn'], 'err': f'{type(e).__name__}: {str(e)[:200]}'})
                print(f"  ⚠ #{c['turn']} {type(e).__name__} → 仲裁队列", flush=True)

        workers = workers_arg(args)
        if workers <= 1:
            for c in todo:
                safe(c)
        else:
            from concurrent.futures import ThreadPoolExecutor
            print(f'  并发 workers={workers}', flush=True)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(safe, todo))

    # Normalise to one row per turn before building the inventory, so that duplicate rows
    # written by concurrent processes are not counted twice.
    # 建库存前先规范为每轮次一行, 避免并发写入的重复行被重复计入。
    kept, dropped = dedup_jsonl_by(res_p, 'turn')
    if dropped:
        print(f'⚠ anchors.jsonl 发现同 turn 多份: 保留最后一份 {kept} 行, 丢弃 {dropped} 行 → anchors_duplicates.jsonl(备查)')
    rows = load_jsonl(res_p)
    inv = build_inventory(rows, os.path.join(d, 'anchor_inventory.json'))
    n_earlier = sum(1 for r in rows for a in r['anchors'] if 'earlier_hit' in a)
    n_badtype = sum(1 for r in rows for a in r['anchors'] if 'type_out_of_domain' in a)
    print(f'P3 汇总: 位置 {len(rows)} · 锚点 {sum(len(r["anchors"]) for r in rows)} '
          f'· 库存 {len(inv)} · earlier_hit 报警 {n_earlier} '
          f'· 漏锚审计非空 {sum(1 for r in rows if r["audit_missing"])} '
          f'· type 越界 {n_badtype}')
    print(f'→ {res_p}')


if __name__ == '__main__':
    main()
