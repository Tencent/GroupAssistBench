#!/usr/bin/env python3
# Shared model-call layer for P2-P5.
#   - reasoning-model compatibility (max_completion_tokens + reasoning_effort, no temperature)
#   - on finish=length, double the output budget and retry
#   - token pacing (PACE_TPM) to stay under per-minute limits
#   - per-call usage accounting in usage.jsonl; run_meta is aggregated by the caller
#   - strict JSON: a parse failure is retried at most twice, then raised so the caller
#     can route the item to the arbitration queue; partial output is never salvaged
# P2–P5 共用的模型调用层: 推理模型兼容、截断时加倍预算重试、配速、逐调用记账、严格 JSON 解析。
import fcntl, json, os, re, threading, time

_IO_LOCK = threading.Lock()  # shared by worker threads: appends to usage.jsonl and artifacts must not interleave

# Endpoint profile: a custom OPENAI_BASE_URL is assumed to be a self-hosted or proxied
# gateway without a per-minute token cap, so pacing and concurrency defaults are raised.
# Leaving it unset assumes a direct vendor key with a modest TPM quota.
# Explicit PACE_TPM / --workers always win over these defaults.
# 设了 OPENAI_BASE_URL 视为自建/代理网关 → 放开配速与并发; 未设视为官方直连 → 保守默认。
# 显式 PACE_TPM / --workers 优先。
# An empty value is not "unset" for the SDK: `source .env` exports OPENAI_BASE_URL="" and
# the client would use it as the base URL. Drop it so that an empty line really means the
# official endpoint.
# 空值对 SDK 不等于未设置: source .env 会导出空串, 这里把它去掉, 空值即表示官方端点。
if not os.environ.get('OPENAI_BASE_URL'):
    os.environ.pop('OPENAI_BASE_URL', None)
CUSTOM_ENDPOINT = bool(os.environ.get('OPENAI_BASE_URL'))
INTERNAL = CUSTOM_ENDPOINT  # alias kept for older callers
MODEL_MAIN = os.environ.get('MODEL_MAIN', 'gpt-5.5')  # main model: locating, anchoring, classification, item writing, re-judging
MODEL_CHEAP = os.environ.get('MODEL_CHEAP', 'gpt-5.5')  # first-pass request check and the counterfactual probe in P2
PACE_TPM = int(os.environ.get('PACE_TPM', '200000' if INTERNAL else '25000'))
PACE = PACE_TPM / 60.0
DEFAULT_WORKERS = 8 if INTERNAL else 1
REASONING_EFFORT = os.environ.get('REASONING_EFFORT', 'minimal')


def code_commit():
    """Pipeline code version (short git HEAD sha, '+dirty' if the tree has uncommitted
    changes), recorded in run_meta so every run can be traced to the code that made it.
    代码版本(git 短 sha), 写入 run_meta 以便追溯产物来自哪版代码。
    """
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=5).stdout.strip() or 'unknown'
        dirty = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--', 'pipeline', 'tools'],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return sha + ('+dirty' if dirty else '')
    except Exception:
        return 'unknown'


def config_banner():
    """Print the effective configuration at start-up; run_meta carries the same fields.
    Only "custom" / "default" is printed, never the base URL itself, because this line
    is often pasted into issues.
    启动时打印生效配置; 只打印 custom/default, 不打印端点地址。
    """
    return (f"[llm] endpoint={'custom' if CUSTOM_ENDPOINT else 'default'} "
            f"main={MODEL_MAIN} cheap={MODEL_CHEAP} pace_tpm={PACE_TPM} "
            f"workers_default={DEFAULT_WORKERS} ctx_full={os.environ.get('CTX_FULL', '0')} "
            f"ctx_max={os.environ.get('CTX_MAX_TOKENS', '120000')} force_json={os.environ.get('FORCE_JSON', '1')}")


def is_reasoning(model):
    m = model.lower()
    return m.startswith('gpt-5') or m.startswith('o1') or m.startswith('o3') or m.startswith('o4')


# Some gateways wrap transient failures in HTTP 400 (e.g. "downstream service temporarily
# unavailable"). The SDK treats 400 as non-retryable, so a short outage would abort a
# whole stage and lose its unsaved output. Only 400s that the gateway itself marks as
# transient are retried here; genuine request errors and content refusals still raise.
# 部分网关把临时故障也包成 HTTP 400; 这里只对网关自述可重试的 400 做退避重试, 其余照常抛出。
_RETRYABLE_400 = ('-960000', '下游服务暂不可用', '请稍后重试', 'service_offline', 'temporarily unavailable')


def _is_transient(e):
    from openai import APIStatusError
    if not isinstance(e, APIStatusError):
        return False
    if e.status_code in (429, 500, 502, 503, 504):
        return True
    if e.status_code == 400:
        return any(k in str(e) for k in _RETRYABLE_400)
    return False


def chat(out_dir, stage, sys_p, user_p, model=None, max_tokens=600, force_json=False):
    """One call. On finish=length the budget is doubled and the call retried. Returns the
    text, possibly empty; the parser above decides whether it is a failure.
    force_json sets response_format=json_object: a prompt alone does not stop smaller
    models from answering the chat instead of returning the requested JSON.
    FORCE_JSON=0 turns this off globally for endpoints that do not support it.
    单次调用: 截断时加倍预算重试; force_json 在 API 层强制 JSON 输出, FORCE_JSON=0 可关闭。
    """
    from openai import OpenAI
    cli = OpenAI()
    model = model or MODEL_MAIN
    is_r = is_reasoning(model)
    cap = 12000 if is_r else 3200
    cur = max(max_tokens, 4000) if is_r else max_tokens
    fj = force_json and os.environ.get('FORCE_JSON', '1') == '1'
    for _ in range(4):
        if is_r:
            kwargs = dict(model=model, max_completion_tokens=cur, reasoning_effort=REASONING_EFFORT,
                          messages=[{'role': 'system', 'content': sys_p},
                                    {'role': 'user', 'content': user_p}])
        else:
            kwargs = dict(model=model, temperature=0, max_tokens=cur,
                          messages=[{'role': 'system', 'content': sys_p},
                                    {'role': 'user', 'content': user_p}])
        if fj:
            kwargs['response_format'] = {'type': 'json_object'}
        t0 = time.time()
        for _a in range(6):  # transient gateway errors: back off 1/2/4/8/16 s (see _is_transient)
            try:
                r = cli.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if _a == 5 or not _is_transient(e):
                    raise
                time.sleep(2 ** _a)
                print(f'⚠ {stage} 网关瞬时故障, {2 ** _a}s 后第 {_a + 1} 次重试: '
                      f'{str(e)[:120]}', flush=True)
        lat = round(time.time() - t0, 1)  # model latency only, excluding pacing sleeps
        u = r.usage
        fr = getattr(r.choices[0], 'finish_reason', '?')
        rt = getattr(getattr(u, 'completion_tokens_details', None), 'reasoning_tokens', 0)
        with _IO_LOCK, open(os.path.join(out_dir, 'usage.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'stage': stage, 'model': model, 'in': u.prompt_tokens,
                                'out': u.completion_tokens, 'reasoning': rt,
                                'finish': fr, 'budget': cur, 'lat_s': lat}) + '\n')
        time.sleep(max(1.0, u.prompt_tokens / PACE))
        content = r.choices[0].message.content or ''
        if fr == 'length' and cur < cap:
            cur = min(cur * 2, cap)
            print(f'⚠ {stage} truncated → 预算加倍至 {cur} 重试', flush=True)
            continue
        return content
    return ''


class ParseFail(Exception):
    pass


def json_obj(text):
    m = re.search(r'\{.*\}', text or '', re.S)
    if not m:
        raise ParseFail(f'无 JSON 对象: {(text or "")[:120]!r}')
    return json.loads(m.group())


def json_arr(text):
    m = re.search(r'\[.*\]', text or '', re.S)
    if not m:
        raise ParseFail(f'无 JSON 数组: {(text or "")[:120]!r}')
    return json.loads(m.group())


def chat_json(out_dir, stage, sys_p, user_p, model=None, max_tokens=600, arr=False, retries=2):
    """Strict JSON call: parse failures are retried up to `retries` times, then ParseFail
    is raised (the caller records it in the arbitration queue).
    In json_object mode an array must be wrapped in an object; callers with arr=True
    already ask for that shape in their prompts.
    严格 JSON 调用: 解析失败重试 ≤retries 次, 仍失败抛 ParseFail。数组需包在对象里返回。
    """
    last = None
    for i in range(retries + 1):
        out = chat(out_dir, stage, sys_p, user_p, model=model, max_tokens=max_tokens,
                   force_json=not arr)
        try:
            return (json_arr if arr else json_obj)(out)
        except (ParseFail, json.JSONDecodeError) as e:
            last = e
            print(f'⚠ {stage} 解析失败({i + 1}/{retries + 1}): {e}', flush=True)
    raise ParseFail(str(last))


_NORM_STRIP = re.compile(r'[\s*_`~＊]+')

def norm_text(s):
    """Normalise before literal matching: drop whitespace and markdown markers, because
    models often quote **bold** text without the asterisks. Characters are only removed,
    never reordered.
    字面比对前的归一化: 去掉空白与 markdown 修饰符; 只删字符, 不改变次序。
    """
    return _NORM_STRIP.sub('', s or '')


def span_in(span, text):
    return bool(span) and norm_text(span) in norm_text(text)


def append_jsonl(path, row):
    with _IO_LOCK, open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def append_if_absent(path, row, key='turn'):
    """Append a row unless its key already exists; safe across processes. Returns True if
    written, False if the key was already there.
    A thread lock alone cannot stop two processes running the same stage from both
    writing the same key, so the file is re-read under flock and the check and the write
    happen in one critical section.
    按 key 去重的追加写, 跨进程安全: 在文件锁内重读已有 key 再决定是否写入。
    """
    lock_p = path + '.lock'
    with _IO_LOCK, open(lock_p, 'a+') as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            existing = set()
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    for ln in f:
                        if ln.strip():
                            try:
                                existing.add(json.loads(ln).get(key))
                            except json.JSONDecodeError:
                                pass  # partial or corrupt lines are ignored here and cleaned up by dedup
            if row.get(key) in existing:
                return False
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            return True
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)


def dedup_jsonl_by(path, key='turn', dup_suffix='_duplicates'):
    """Normalise a jsonl file to one row per key, keeping the LAST copy, and write the
    dropped rows to <name>_duplicates.jsonl instead of deleting them silently.
    Keeping the last copy does not change downstream results: readers build
    {key: row} dicts, in which the last copy already wins.
    Returns (rows kept, rows dropped).
    把 jsonl 规范为每个 key 一行, 保留最后一份, 被丢弃的写入 _duplicates.jsonl。返回(保留数, 丢弃数)。
    """
    rows = load_jsonl(path)
    keep, dropped = {}, []
    for r in rows:
        k = r.get(key)
        if k in keep:
            dropped.append(keep[k])
        keep[k] = r
    if not dropped:
        return len(rows), 0
    base, ext = os.path.splitext(path)
    with open(base + dup_suffix + ext, 'a', encoding='utf-8') as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for k in sorted(keep, key=lambda x: (x is None, x)):
            f.write(json.dumps(keep[k], ensure_ascii=False) + '\n')
    os.replace(tmp, path)
    return len(keep), len(dropped)


def workers_arg(args, default=None):
    """--workers N: per-candidate concurrency (candidates are independent; appends are locked).
    Pacing is per thread, so the total rate is about N x PACE_TPM: keep 1 for a personal
    key; with a gateway and PACE_TPM=200000, 4-8 is typical.
    候选级并发。配速按线程计, 总速率约为 N×PACE_TPM。
    """
    if default is None:
        default = DEFAULT_WORKERS
    return int(args[args.index('--workers') + 1]) if '--workers' in args else default


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
