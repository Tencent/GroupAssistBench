#!/usr/bin/env python3
# P2 Locate requests (paper 3.2 "locating requests"): enumerate messages addressed to
# the assistant, drop non-requests, then answer each request twice - with and without
# the preceding history - and keep it only if the history changes the answer.
#   P2-1 enumerate (code)   each bot reply is traced back to the human message(s) that
#                           triggered it; human messages that @ the assistant but got no
#                           reply are collected too
#   P2-2 is_request (model) True -> RT1 material; False followed by a user correction ->
#                           RT3 signal (recorded only, not built); otherwise dropped
#   P2-3 grounding (model)  answer once with and once without the group history, then a
#                           comparator decides whether the history matters; bot-generated
#                           content does not count as group memory
# Output: candidates.jsonl {turn, query, src_turn, is_request, route, grounding{v,by,evidence}, ...}
# Usage: python3 pipeline/p2_locate.py <gid> [--enum-only] [--use-gold-grounding] [--limit N] [--turns a,b] [--workers N]
#   --enum-only           run only the zero-API enumeration (P2-1)
#   --use-gold-grounding  where human annotations exist, take their grounding / is_request
# P2 定位: 枚举向助手发出的消息, 去掉非请求, 再分别在有/无群历史的条件下作答, 只保留依赖历史的请求。
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.llm import (config_banner, MODEL_CHEAP, MODEL_MAIN, append_jsonl, append_if_absent,
                         chat, chat_json, load_jsonl, workers_arg, ParseFail)
from pipeline.contract import ROOT
from pipeline.examples import render as render_examples

AT_BOT = re.compile(r'@元宝|@Yuanbao', re.I)


def rundir(gid):
    env = os.environ.get('PIPE_RUN_DIR')
    if env:
        return env if os.path.isabs(env) else os.path.join(ROOT, 'pipeline_runs', env)
    # isdir is required: input files kept under pipeline_runs/ (e.g. <gid>_raw.jsonl) must
    # not be mistaken for run directories.
    # 必须判断 isdir, 否则 pipeline_runs/ 下的输入文件会被当成 run 目录。
    base = os.path.join(ROOT, 'pipeline_runs')
    cands = sorted(d for d in os.listdir(base)
                   if d.startswith(gid + '_') and os.path.isdir(os.path.join(base, d)))
    if not cands:
        raise SystemExit(f'未找到 pipeline_runs/{gid}_*/ , 先跑 pipeline/ingest.py {gid}')
    return os.path.join(ROOT, 'pipeline_runs', cands[-1])


def load_msgs(d):
    return load_jsonl(os.path.join(d, 'messages.jsonl'))


# ---------------- P2-1 enumerate service points (code) ----------------

BLOCK_MAX = int(os.environ.get('ENUM_BLOCK_MAX', '5'))  # maximum length of a same-speaker block
THIN_QUERY_MAX = int(os.environ.get('THIN_QUERY_MAX', '10'))  # a net query (after removing @bot) this short or shorter is not self-contained


PUNCT_PAT = re.compile(r'[\s，。！？、,.!?~～:：;；"\'“”‘’()（）\[\]【】…·—\-]+')


def net_query_len(q):
    """Net query length: characters left after removing the @bot mention, punctuation and
    whitespace. Punctuation carries no content and does not count.
    净题面长度: 去掉 @bot、标点与空白后剩余的实义字符数。
    """
    return len(PUNCT_PAT.sub('', AT_BOT.sub('', q or '')))


def is_thin_query(q):
    """Is the query not self-contained, i.e. almost nothing is left after removing @bot?
    Such positions are either a continuation (the real request is in the same speaker's
    previous message; handled by block grouping) or pure social chatter with nothing to
    do. This deterministic check stops the ones still thin after grouping before P3, the
    most expensive stage, and routes them to human review instead of building them.
    题面是否不自足。这类位置要么是承接式(真请求在同一发言人的上一条, 已由块化处理),
    要么是纯社交。此处在 P3 之前拦下仍不自足的位置, 送人工分流。
    """
    return net_query_len(q) <= THIN_QUERY_MAX


def enumerate_candidates(msgs):
    """Service-point coordinates (same convention as the human annotations): candidate turn =
    the bot's service turn (the first of consecutive bot turns); src_turn = the human turn
    that triggered it; query = the same speaker's consecutive block of messages ending at
    src_turn. Human messages that @ the assistant but got no reply are also collected
    (turn = the human turn, no service).
    The query is a same-speaker block because a request is often split across messages
    (an image or file first, then "@assistant"); taking only the last message would leave
    an empty query. Other speakers' messages are not included, which keeps blocks short.
    The whole block is registered as taken, so a block that starts with @assistant is not
    collected a second time as an unanswered request.
    服务点坐标: turn 为 bot 服务行, src_turn 为触发它的人类行, query 为该发言人连续发出的消息块。
    """
    out, taken_src = [], set()
    for i, m in enumerate(msgs):
        if m['kind'] != 'bot':
            continue
        if i > 0 and msgs[i - 1]['kind'] == 'bot':
            continue  # consecutive bot turns are one service
        j = next((j for j in range(i - 1, -1, -1) if msgs[j]['kind'] != 'bot'), None)
        if j is None:
            continue
        q = msgs[j]
        b = j  # walk back over the same speaker's consecutive human messages
        while (b - 1 >= 0 and msgs[b - 1]['kind'] != 'bot'
               and msgs[b - 1]['who'] == q['who'] and j - (b - 1) + 1 <= BLOCK_MAX):
            b -= 1
        blk = msgs[b:j + 1]
        taken_src.update(x['turn'] for x in blk)  # register the whole block so the second loop does not collect it again
        out.append({'turn': m['turn'],
                    'query': '\n'.join(x['text'] or '' for x in blk),
                    'who': q['who'],
                    'src_turn': q['turn'],  # the last message of the block triggered the bot; cut_turn semantics unchanged
                    'service_turn': m['turn'],
                    'block_turns': [x['turn'] for x in blk],
                    'enum_by': 'rule:bot_backtrack' + ('_block' if len(blk) > 1 else '')})
    for m in msgs:
        if m['kind'] != 'bot' and AT_BOT.search(m['text'] or '') and m['turn'] not in taken_src:
            out.append({'turn': m['turn'], 'query': m['text'], 'who': m['who'],
                        'src_turn': m['turn'], 'service_turn': None,
                        'block_turns': [m['turn']],
                        'enum_by': 'rule:at_mention_unanswered'})
    return sorted(out, key=lambda x: x['turn'])


# ---------------- P2-2 is_request (cheap model) ----------------
# The criterion is whether there is something still to be done, not whether the assistant
# was @-mentioned; agreement and small talk are given as explicit negatives.
# 判据是「有没有一件还没做的事」, 而不是「有没有 @ 助手」; 附和与寒暄作为显式反例。
REQ_SYS = """你是数据标注流水线中的「请求形态」判定器。给定群聊里一条被 AI 助手回应过(或 @ 了助手)的人类消息, 判定它是不是一个「向助手要服务的真请求」。你只做分类, **绝对不要回答消息本身的内容**。

# 核心判据(先问这一句)
这条消息里**是否存在一件还没被做的事**? 有 → true; 只是社交表达 → false。
**@ 了助手不等于提了请求。**

# is_request=true
向助手提出的提问/指令/求助 —— 助手不回应就有事情没被完成。
真实用户大量使用非规范表达, 这些仍算请求:
  · 情绪化追问(「？？红红？！」—— 在追问某个结果)
  · 引用+质疑(「我唱的是X, 你是不是没看」)
  · 命令式(「发广告直接清走」)
  · 承接上文(「[跑步打卡图]\n@元宝 刚跑完」—— 待办是「记录这次打卡」)

# is_request=false
  · **附和 / 寒暄 / 情绪表达 —— @了助手但没有待办**
    (「还行吧@元宝」「我赞同元宝说的」「哇, 好好看, 我也想要@元宝」「OK, 接着肝喽」「？？？！！！」)
  · 写入设定(「记住我们都是干啥的」)
  · 与助手无关的成员闲聊
  · 纯陈述(无回应期待)

# 边界提示
助手事后回应了, **不能倒推出**这是请求 —— 助手会主动接寒暄的话。只看这条消息本身有没有待办。

query_form: request(能独立读懂的规范请求) | utterance(非规范表达但仍构成请求)

# 输出严格 JSON(只输出这一个对象)
{"is_request":true,"query_form":"request|utterance","reason":"≤15字"}"""

CORRECT_PAT = re.compile(r'不对|错了|不是这样|记错|说的是|纠正|重新|搞错|弄错|别乱|瞎说')


def has_followup_correction(msgs, service_turn, window=6):
    """RT3 signal (code, conservative): a human correction or protest within `window`
    messages after the bot's service.
    RT3 信号: bot 服务后若干条内出现人类的纠正或抗议。
    """
    if service_turn is None:
        return False
    after = [m for m in msgs if m['turn'] > service_turn and m['kind'] != 'bot'][:window]
    return any(CORRECT_PAT.search(m['text'] or '') for m in after)


# ---------------- P2-3 grounding counterfactual ----------------
# Bot-generated content is excluded from group memory in these prompts.
ANS_A = ('你是群聊AI助手。下面给出本群成员档案与聊天记录。简要回答最后的提问；'
         '凡用到群内已知信息，直接给出具体值(数值/人名/约定/进度)。中文，不超过120字。')
ANS_B = ('你是群聊AI助手。你没有该群的任何历史记录。简要回答提问；信息不足时按常识给出最可能的回答。'
         '中文，不超过120字。')
CMP_SYS = """你是数据标注流水线中的「grounding 三分」比较器。同一提问有两份回答：A(可见群历史+成员档案) 与 B(无任何历史)。按反事实判据分类。

# 判据(钉死)
否: B 与 A 实质等效——通用知识/常识即可答好, 群历史没带来实质差别
弱: A 仅在个性化包装上更好(称呼/语气/贴场景), 实质内容与 B 相同
强: A 含 B 没有且无法凭常识补出的群内特异内容(成员的决定/偏好/约定/进度/具体数值), 缺之即答错或答空

# 排除规则: 「群记忆」只指成员产生的内容(成员的发言/决定/创作/纠正)。
# bot 自己生成过的知识回答/方案/人设解释不算群记忆——A 引用了 bot 早前的回答而 B 没有, 不构成「强」。

# 输出严格 JSON
{"grounding":"强|弱|否","evidence":"A中来自群历史的关键事实,≤30字,无则空串","reason":"≤25字"}"""


CTX_MAX_TOKENS = int(os.environ.get('CTX_MAX_TOKENS', '120000'))  # context budget; set about 26000 for a personal key with a ~30k per-request limit


def build_ctx(msgs, upto, profile, ctx_full=None):
    """Build the context (P3-1). With CTX_FULL=1 the full history is used; if its estimate
    exceeds CTX_MAX_TOKENS (about 1.5 Chinese characters per token), fall back to layered
    truncation (the latest 100 messages in full, older ones cut to 40 characters), marked
    at the start of each line.
    上下文: CTX_FULL=1 时给全文; 超过 CTX_MAX_TOKENS 时退回分层截断(近 100 条全文, 更早的截到 40 字)。
    """
    ctx_full = (os.environ.get('CTX_FULL', '') == '1') if ctx_full is None else ctx_full
    hist = [m for m in msgs if m['turn'] < upto]
    if ctx_full and sum(len(m['text'] or '') for m in hist) / 1.5 > CTX_MAX_TOKENS:
        ctx_full = False
    lines = []
    for i, m in enumerate(hist):
        if ctx_full:
            cap = 10 ** 9
        elif len(hist) - i <= 100:
            cap = 80 if m['kind'] == 'bot' else 200
        else:
            cap = 40
        lines.append(f"#{m['turn']} [{m['who']}] {(m['text'] or '')[:cap]}")
    # Profile entries are also cut at the position: only entries established before `upto`
    # are shown, otherwise the context would reveal facts from the future. Entries without
    # an int turn are excluded because they cannot be shown to precede `upto`.
    # 档案条目同样按位置裁剪, 只给此前确立的条目, 防止看到未来; turn 非整数的条目一并排除。
    prof_upto = [e for e in profile if isinstance(e.get('turn'), int) and e['turn'] < upto]
    pb = '===成员档案(fact + 首次出现turn)===\n' + '\n'.join(
        f"- [{e['member']}·{e['slot']}] {e['fact']} (#{e['turn']})" for e in prof_upto) + '\n'
    return pb + '===群聊记录(#序号越大越晚)===\n' + '\n'.join(lines) + '\n===记录结束===\n'


def cf_probe(d, msgs, profile, cand):
    ctx = build_ctx(msgs, cand['turn'], profile)
    ans_a = chat(d, 'p2_ansA', ANS_A, ctx + f"\n提问(#{cand['turn']}): {cand['query']}",
                 model=MODEL_CHEAP, max_tokens=300)
    ans_b = chat(d, 'p2_ansB', ANS_B, f"提问: {cand['query']}", model=MODEL_CHEAP, max_tokens=300)
    v = chat_json(d, 'p2_cmp', CMP_SYS,
                  f"提问: {cand['query']}\n\n回答A(有历史):\n{ans_a}\n\n回答B(无历史):\n{ans_b}",
                  model=MODEL_MAIN, max_tokens=600)
    return {'v': v.get('grounding'), 'evidence': v.get('evidence', ''),
            'reason': v.get('reason', ''), 'by': f'probe:cf({MODEL_CHEAP}→{MODEL_MAIN})',
            'ans_a': ans_a, 'ans_b': ans_b}


# ---------------- main ----------------

def main():
    args = sys.argv[1:]
    gid = next((a for a in args if not a.startswith('-')), 'demo')
    enum_only = '--enum-only' in args
    use_gold = '--use-gold-grounding' in args
    limit = int(args[args.index('--limit') + 1]) if '--limit' in args else 0
    turns = ({int(x) for x in args[args.index('--turns') + 1].split(',')}
             if '--turns' in args else None)  # re-run selected service points
    print(config_banner(), flush=True)
    d = rundir(gid)
    msgs = load_msgs(d)
    profile = json.load(open(os.path.join(d, 'profile.json'), encoding='utf-8')) \
        if os.path.exists(os.path.join(d, 'profile.json')) else []
    gold = {a['turn']: a for a in load_jsonl(os.path.join(d, 'annotations_gold.jsonl'))}

    cands = enumerate_candidates(msgs)
    res_p = os.path.join(d, 'candidates.jsonl')
    done = {r['turn'] for r in load_jsonl(res_p)}
    todo = [c for c in cands if c['turn'] not in done]
    if turns is not None:
        todo = [c for c in todo if c['turn'] in turns]
    if limit:
        todo = todo[:limit]
    print(f'P2: 服务点 {len(cands)} · 已完成 {len(done)} · 本轮 {len(todo)}'
          f'{" (enum-only)" if enum_only else ""}', flush=True)

    req_sys = REQ_SYS + render_examples('is_request', exclude_gid=gid)

    def process(c):
        """Process one candidate (thread-safe: append-only writes, serialised by llm._IO_LOCK)."""
        g = gold.get(c['turn'])
        # P2-2 request check
        if use_gold and g and g['labels'].get('is_request', {}).get('v') is not None:
            c['is_request'] = {'v': bool(g['labels']['is_request']['v']),
                               'by': g['labels']['is_request']['by']}
            c['query_form'] = 'request'
        elif enum_only:
            c['is_request'] = {'v': None, 'by': 'pending'}
        else:
            try:
                msg = f"<待判定消息 turn={c['turn']} 发言人={c['who']}>\n{c['query']}\n</待判定消息>"
                v = chat_json(d, 'p2_req', req_sys, msg, model=MODEL_CHEAP, max_tokens=200)
                by = f'model:cheap({MODEL_CHEAP})'
                first = {'v': bool(v.get('is_request')), 'reason': str(v.get('reason', ''))[:40]}
                # Second opinion from the main model, in both directions:
                #   judged False but the assistant was @-mentioned -> a real request may have been missed
                #   judged True but the query is not self-contained -> small talk may have been taken
                #                                                      for a request
                # 主力模型双向复判: 判否但 @ 了助手 → 可能漏判; 判是但题面不自足 → 可能误判。
                if ((not v.get('is_request') and AT_BOT.search(c['query'] or ''))
                        or (v.get('is_request') and is_thin_query(c['query']))):
                    v = chat_json(d, 'p2_req2', req_sys, msg, model=MODEL_MAIN, max_tokens=200)
                    by = f'model:cheap→main({MODEL_MAIN})'
                # Keep the model's reason so that dropped positions can be reviewed later. When a second
                # opinion was taken, the cheap model's verdict is stored too, so disagreements stay visible.
                # 保存判定理由, 便于事后复核被丢弃的位置; 经过复判的同时保留初判结论。
                c['is_request'] = {'v': bool(v.get('is_request')), 'by': by,
                                   'reason': str(v.get('reason', ''))[:40]}
                if '→main' in by:
                    c['is_request']['cheap'] = first
                c['query_form'] = v.get('query_form', 'request')
            except ParseFail as e:
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P2-2', 'turn': c['turn'], 'err': str(e)})
                return None
        # Route: RT3 is recorded as a signal only and not built.
        if c['is_request']['v']:
            c['route'] = 'RT1'
        elif c['is_request']['v'] is False and has_followup_correction(msgs, c['service_turn']):
            c['route'] = 'RT3_signal'
        elif c['is_request']['v'] is False:
            c['route'] = 'drop'
        else:
            c['route'] = 'pending'
        # Judged a request but not self-contained -> not built directly; sent to human review.
        # 判为请求但题面不自足 → 不直接出题, 送人工分流。
        if c['route'] == 'RT1' and is_thin_query(c['query']):
            c['route'] = 'thin_query'
            c['thin_net_len'] = net_query_len(c['query'])
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P2-4', 'turn': c['turn'], 'src_turn': c.get('src_turn'),
                          'q': (c.get('query') or '')[:60], 'net_len': c['thin_net_len'],
                          'block_turns': c.get('block_turns'),
                          'reason': f'thin_query(净题面≤{THIN_QUERY_MAX}字, 需人判: 承接式漏收 or 纯社交)'})
        # P2-3 grounding
        if use_gold and g and g['labels'].get('grounding', {}).get('v'):
            c['grounding'] = {'v': g['labels']['grounding']['v'], 'by': g['labels']['grounding']['by']}
        elif enum_only or c['route'] in ('drop', 'pending'):
            c['grounding'] = {'v': None, 'by': 'pending'}
        else:
            try:
                c['grounding'] = cf_probe(d, msgs, profile, c)
            except ParseFail as e:
                append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                             {'stage': 'P2-3', 'turn': c['turn'], 'err': str(e)})
                return None
        append_if_absent(res_p, c)  # deduplicated across processes
        print(f"  #{c['turn']:4d} route={c['route']:10s} grounding={c['grounding']['v']}", flush=True)

        return c

    def safe(c):
        """Per-candidate error isolation: any exception is recorded in the arbitration queue
        and the batch continues.
        候选级错误隔离: 任何异常都记入仲裁队列, 批次继续。
        """
        try:
            return process(c)
        except Exception as e:
            append_jsonl(os.path.join(d, 'arbitration_queue.jsonl'),
                         {'stage': 'P2', 'turn': c['turn'], 'err': f'{type(e).__name__}: {str(e)[:200]}'})
            print(f"  ⚠ #{c['turn']} {type(e).__name__} → 仲裁队列", flush=True)

    workers = workers_arg(args)
    if workers <= 1:
        for c in todo:
            safe(c)
    else:
        from concurrent.futures import ThreadPoolExecutor
        print(f'  并发 workers={workers}(总 TPM≈{workers}×PACE_TPM)', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(safe, todo))
    rows = load_jsonl(res_p)
    from collections import Counter
    print('P2 汇总:', dict(Counter(r['route'] for r in rows)),
          '| grounding:', dict(Counter(str(r['grounding']['v']) for r in rows)))
    print(f'→ {res_p}')


if __name__ == '__main__':
    main()
