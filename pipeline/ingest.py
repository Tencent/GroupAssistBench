#!/usr/bin/env python3
# P1 Ingest (JSONL input): a group chat log as JSONL -> normalised timeline, member profile and
# session segments. Pure code except the optional profile extraction.
#   messages.jsonl          unified turn index; `kind` separates bot and human turns
#   profile.json            member profile: model-extracted, every quote verified in code
#   segments.json           sessions split by a time-gap threshold
#   attachments             JSONL logs carry none; bot replies that mention an image
#                           are logged as indirect evidence
#   annotations_gold.jsonl  read-only view of the human annotations
# Usage (package root): python3 pipeline/ingest.py <gid> [--no-profile]
# P1 接入(JSONL 输入): JSONL 群聊记录 → 消息时间线、成员档案、会话分段。除可选的档案抽取外不调模型。
import json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.contract import ROOT, load_annotations
from pipeline.llm import code_commit

ONLINE_MERGED = os.environ.get('PIPE_INPUT_JSONL', os.path.join(ROOT, 'data', 'messages.jsonl'))
MODEL_PROFILE = os.environ.get('MODEL_PROFILE', 'gpt-5.5')
GAP_MIN = int(os.environ.get('SEG_GAP_MIN', '30'))  # session gap threshold (minutes)
IMG_HINT = re.compile(r'从图片看|这张图|图里|看图|图上')


def build_messages(gid):
    if not os.path.exists(ONLINE_MERGED):
        raise SystemExit(f'找不到输入文件: {ONLINE_MERGED}；请设置 PIPE_INPUT_JSONL')
    msgs = []
    for ln, l in enumerate(open(ONLINE_MERGED, encoding='utf-8')):
        if not l.strip():
            continue
        r = json.loads(l)
        raw_gid = str(r.get('group_id', ''))
        if raw_gid != gid and not raw_gid.startswith(gid):
            continue
        text = r.get('content') or ''
        m = {'gid': gid, 'turn': int(r['idx']),
             'who': r['send_uid'], 'text': text,
             'ts': r.get('event_time_iso'),
             'kind': 'bot' if r['send_uid'] == 'bot' else 'text',
             'reply_to': None, 'attachments': [],
             'src': {'track': 'jsonl', 'file': os.path.basename(ONLINE_MERGED), 'row': ln}}
        if 'safe' in str(r.get('rewrite_method', '')):
            m['safety_filtered'] = True
        # JSONL logs have no attachment layer. A bot reply that says it is reading an image
        # is the only indirect sign that a user sent one; it is logged as evidence and used to
        # exclude questions that depend on the image.
        # JSONL 记录无附件; bot 回复中「从图片看」是用户发图的间接证据, 用于排除依赖图片的题。
        if m['kind'] == 'bot' and IMG_HINT.search(text):
            m['attachments'] = [{'type': 'image_evidence', 'note': 'bot 回复引用了图, 图本体不在 trace'}]
        msgs.append(m)
    return sorted(msgs, key=lambda x: x['turn'])


def build_segments(msgs, gap_min=GAP_MIN):
    segs, cur = [], []
    prev = None
    for m in msgs:
        t = time.mktime(time.strptime(m['ts'][:19], '%Y-%m-%dT%H:%M:%S')) if m.get('ts') else None
        if cur and prev and t and (t - prev) > gap_min * 60:
            segs.append(cur)
            cur = []
        cur.append(m['turn'])
        prev = t if t else prev
    if cur:
        segs.append(cur)
    return [{'seg': i, 'turn_from': s[0], 'turn_to': s[-1], 'n': len(s)}
            for i, s in enumerate(segs)]


PROFILE_PROMPT = """你是数据标注流水线中的「群档案抽取」模块。给定一个群的完整时间线，为每位人类成员抽取七类槽位的事实条目。你的输出会被程序逐条校验：每条必须给出 turn 与 quote，quote 必须是该 turn 消息原文的连续子串，校验失败该条即被丢弃。

# 槽位定义
状态(当前处境/近况) · 偏好(喜好/习惯) · 承诺(说过要做的事) · 约束(要求他人或bot遵守的规则) · 事实(身份/属性/经历等稳定事实) · 人设(对bot或成员的角色设定) · 既有方案(bot给过且被采纳的针对性建议)

# 规则
1. 只录与成员个体相关的条目；bot 的通用知识回答不录；「既有方案」只录针对本群成员的具体方案。
2. fact 写具体值，不写指针（✗「他提过一个计划」 ✓「计划拍浩然舞模仿视频」）。
3. 同一事实多次提及，turn 取最早确立点。
4. 宁缺毋滥：拿不准的不录。

# 输出（严格 JSON 数组，不要输出任何其它内容）
[{"member":"user_1","slot":"偏好","fact":"具体值","turn":8,"quote":"该turn原文的连续子串(≤30字)"}]"""


def build_profile(msgs, out_dir):
    from openai import OpenAI
    cli = OpenAI()
    # Bot replies are cut to 120 characters and human turns to 250: member facts live
    # mostly in human turns, and this keeps the request within common rate limits.
    # bot 回复截到 120 字、人类消息截到 250 字: 成员事实主要在人类消息里, 同时控制请求长度。
    lines = [f"#{m['turn']} [{m['who']}] {m['text'][:120 if m['kind']=='bot' else 250]}"
             for m in msgs]
    user_p = '===时间线===\n' + '\n'.join(lines) + '\n===结束==='
    # Output budget: on truncation, double it and retry, up to PROFILE_MAX_TOKENS.
    # Reasoning models count their thinking tokens toward this budget.
    # 输出预算: 被截断就加倍重试, 封顶 PROFILE_MAX_TOKENS。推理模型的思考 token 也计入预算。
    budget = 6000
    cap = int(os.environ.get('PROFILE_MAX_TOKENS', '24000'))
    while True:
        r = cli.chat.completions.create(
            model=MODEL_PROFILE, temperature=0, max_tokens=budget,
            messages=[{'role': 'system', 'content': PROFILE_PROMPT},
                      {'role': 'user', 'content': user_p}])
        u = r.usage
        with open(os.path.join(out_dir, 'usage.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'stage': 'profile', 'model': MODEL_PROFILE, 'budget': budget,
                                'in': u.prompt_tokens, 'out': u.completion_tokens}) + '\n')
        out = r.choices[0].message.content or ''
        fin = r.choices[0].finish_reason
        if fin != 'length' or budget >= cap:
            break
        print(f'⚠ profile 输出被截断(finish=length, budget={budget}) → 预算加倍至 {min(budget * 2, cap)} 重试',
              flush=True)
        budget = min(budget * 2, cap)
    m = re.search(r'\[.*\]', out, re.S)
    if m:
        entries = json.loads(m.group())
    elif fin == 'length':
        # Still truncated at the cap: keep only complete objects and record it explicitly.
        # 到顶仍被截断: 只保留完整的对象, 并显式记录。
        entries = [json.loads(x) for x in re.findall(r'\{[^{}]*\}', out)]
        print(f'⚠ profile 输出到预算上限 {budget} 仍被截断, 抢救到 {len(entries)} 条完整条目')
        if not entries:
            raise RuntimeError(f'profile 抽取失败: 预算 {budget} 用尽仍无完整条目。'
                               f'可调大 PROFILE_MAX_TOKENS 重跑, 或 SKIP_PROFILE=1 显式跳过档案。')
    else:
        raise ValueError(f'profile 输出无 JSON 数组(finish={fin}): {out[:200]!r}')
    # Every quote must be found verbatim in its turn; entries that fail are dropped.
    # 每条 quote 必须能在对应轮次原文中找到, 找不到即丢弃。
    by_turn = {mm['turn']: mm['text'] for mm in msgs}
    ok, rejected = [], []
    for e in entries:
        t = by_turn.get(e.get('turn'))
        if t and e.get('quote') and e['quote'] in t:
            e['by'] = f'model:main({MODEL_PROFILE})'
            e['verified'] = 'quote_match'
            ok.append(e)
        else:
            rejected.append(e)
    return ok, rejected


def main():
    gid = next((a for a in sys.argv[1:] if not a.startswith('-')), 'demo')
    # PIPE_RUN_DIR sets the run directory name; the default is <gid>_<YYYYMMDD>.
    # PIPE_RUN_DIR 指定 run 目录名, 默认 <gid>_<日期>。
    out_dir = os.environ.get('PIPE_RUN_DIR') or f'{gid}_{time.strftime("%Y%m%d")}'
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, 'pipeline_runs', out_dir)
    os.makedirs(out_dir, exist_ok=True)
    msgs = build_messages(gid)
    with open(os.path.join(out_dir, 'messages.jsonl'), 'w', encoding='utf-8') as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    segs = build_segments(msgs)
    json.dump(segs, open(os.path.join(out_dir, 'segments.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    gold = load_annotations(gid=gid)
    with open(os.path.join(out_dir, 'annotations_gold.jsonl'), 'w', encoding='utf-8') as f:
        for a in gold:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    meta = {'gid': gid, 'stage': 'W2', 'n_messages': len(msgs), 'n_segments': len(segs),
            'n_gold': len(gold), 'seg_gap_min': GAP_MIN,
            'profile_model': MODEL_PROFILE, 'code_commit': code_commit(),
            'run_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    if '--no-profile' not in sys.argv:
        ok, rejected = build_profile(msgs, out_dir)
        json.dump(ok, open(os.path.join(out_dir, 'profile.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        if rejected:
            json.dump(rejected, open(os.path.join(out_dir, 'profile_rejected.json'), 'w',
                      encoding='utf-8'), ensure_ascii=False, indent=1)
        meta['profile_entries'] = len(ok)
        meta['profile_rejected'] = len(rejected)
        meta['profile_verify_rate'] = round(len(ok) / max(len(ok) + len(rejected), 1), 3)
    json.dump(meta, open(os.path.join(out_dir, 'run_meta.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f'→ {out_dir}/')


if __name__ == '__main__':
    main()
