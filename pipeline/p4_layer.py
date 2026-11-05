#!/usr/bin/env python3
"""P4 task level: decide L1 (memory recall) vs L2 (memory use) for each position.

The level is a property of the expected answer, not of the question's wording. The
model looks at the answer produced with the full group history (grounding.ans_a, already
stored in candidates.jsonl by the P2 counterfactual probe) and answers one concrete
question: once the checkable facts are removed from this answer, is what remains still
the main thing the asker wanted? If not, the answer is mostly recall (L1); if so, it is
use of memory (L2). This costs one short call per position (question + answer).

Output: the `layer` field of classify.jsonl is updated in place (pending entries only,
idempotent). Medium- and low-confidence verdicts are also sent to arbitration_queue.
Usage: python3 pipeline/p4_layer.py <gid> [--limit N] [--dry-run] [--workers N]

P4 判 L1/L2: 依据「能看到全部历史的回答」本身判断——去掉可核对的事实后, 剩下的
是否仍是提问者要的主要内容。只填 pending 项; 中/低置信度送仲裁队列。
"""
import json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.llm import (config_banner, MODEL_MAIN, append_jsonl, append_if_absent,
                          chat_json, load_jsonl, workers_arg, ParseFail)
from pipeline.p2_locate import rundir

SYS = """你是评测题的「判分方式」判定器。给你一道群聊里真实发生的提问，以及一份**能看到全部群历史**的助手回答。你只判定这道题该用哪种方式判分，**不要评价回答好坏，也不要自己作答**。

# 两种判分方式
L1 = 答案是确定的。可以写成一组事实要点，逐条核对对错（判分 = 数命中了几条）。
L2 = 答案不确定。只能看覆盖了哪些要点、建议是否贴合这个群/这个人（判分 = 按细则评覆盖度与质量）。

# 判据（对着给出的那份回答问，不要根据题面猜）
把这份回答里「可以核对对错的事实」抽掉，剩下的部分是什么？
  · 剩得很少 —— 回答基本就是一组事实/数值/名单 → L1
  · 剩下的是建议、取舍、方案，而且**那才是提问者真正要的** → L2

# 注意
· 混合型很常见（前半是事实、后半是建议）。判准是**提问者要的主要是哪一半**。
· 回答长不代表 L2。一份长清单只要每条都能核对，仍是 L1。
· 回答里模型自己编的建议数值（如"建议分配3万"）不算可核对事实。

# 输出严格 JSON
{"layer":"L1|L2","confidence":"高|中|低","why":"≤15字"}"""


def judge(d, query, ans_a):
    u = f'提问:\n{query}\n\n能看到全部群历史的回答:\n{ans_a}'
    v = chat_json(d, 'p4_layer', SYS, u, max_tokens=200)
    lay = (v.get('layer') or '').strip().upper()
    if lay not in ('L1', 'L2'):
        raise ParseFail(f'layer 取值域外: {v.get("layer")!r}')
    return lay, (v.get('confidence') or '').strip(), (v.get('why') or '').strip()


def main():
    args = sys.argv[1:]
    gid = next((a for a in args if not a.startswith('-')), None)
    if not gid:
        raise SystemExit(__doc__)
    dry = '--dry-run' in args
    limit = int(args[args.index('--limit') + 1]) if '--limit' in args else 0
    print(config_banner(), flush=True)
    d = rundir(gid)
    cls_p = os.path.join(d, 'classify.jsonl')
    rows = load_jsonl(cls_p)
    cands = {c['turn']: c for c in load_jsonl(os.path.join(d, 'candidates.jsonl'))}

    todo, skip = [], Counter()
    for r in rows:
        if (r.get('layer') or {}).get('v'):
            skip['已有layer'] += 1
            continue
        c = cands.get(r['turn'])
        if c is None:
            skip['无候选行'] += 1
        elif c.get('route') != 'RT1':
            skip[f"route={c.get('route')}"] += 1
        elif (c.get('grounding') or {}).get('v') != '强':
            skip['grounding非强'] += 1
        elif not (c.get('grounding') or {}).get('ans_a'):
            skip['无ans_a(不猜, 留pending)'] += 1
        elif r['turn'] not in todo:
            todo.append(r['turn'])  # deduplicate by turn: older artifacts may contain repeated rows
    if limit:
        todo = todo[:limit]
    print(f'P4-layer: 待判 {len(todo)} · 跳过 {dict(skip)}', flush=True)

    filled, dist, conf = {}, Counter(), Counter()

    def process(t):
        c = cands[t]
        try:
            lay, cf, why = judge(d, c.get('query', ''), c['grounding']['ans_a'])
        except ParseFail as e:
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P4-layer', 'turn': t, 'reason': f'layer_unresolved({e})'})
            skip['判不动→仲裁'] += 1
            return
        except Exception as e:                       # noqa: BLE001
            # Any error on one position, not only ParseFail, sends that position to the arbitration
            # queue; the other positions are still written back.
            # 单个位置出任何异常都送仲裁队列, 其余位置照常写回。
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P4-layer', 'turn': t,
                          'reason': f'layer_call_failed({type(e).__name__}: {str(e)[:200]})'})
            skip[f'调用失败→仲裁({type(e).__name__})'] += 1
            return
        filled[t] = {'v': lay, 'by': f'model:layer_from_answer({MODEL_MAIN})',
                     'confidence': cf, 'why': why, 'model': MODEL_MAIN}
        dist[lay] += 1
        conf[cf] += 1
        # Medium- and low-confidence verdicts both go to review.
        # 中、低置信度一律送复核。
        if cf in ('中', '低'):
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P4-layer', 'turn': t, 'layer': lay, 'confidence': cf,
                          'why': why, 'reason': f'layer_confidence_{cf}(已填但需人工复核)'})

    workers = workers_arg(args)
    if workers <= 1:
        for i, t in enumerate(todo, 1):
            process(t)
            if i % 20 == 0:
                print(f'  …{i}/{len(todo)}', flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor
        print(f'  并发 workers={workers}', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(process, todo))

    print(f'\nlayer 分布: {dict(dist)} · 置信度: {dict(conf)}')
    n_review = sum(v for k, v in conf.items() if k in ('中', '低'))
    print(f'需人工复核(中/低置信): {n_review}/{len(filled)}')
    if dry:
        print('--dry-run: 未写回')
        return
    if not filled:
        print('无可填充位置, 不改动文件')
        return
    tmp = cls_p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for r in rows:  # idempotent: edits the current file, never reloads from .bak
            if r['turn'] in filled and not (r.get('layer') or {}).get('v'):
                r['layer'] = filled[r['turn']]
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, cls_p)
    print(f'→ {cls_p}（填充 {len(filled)} 行）')


if __name__ == '__main__':
    main()
