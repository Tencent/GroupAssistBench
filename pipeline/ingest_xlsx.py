#!/usr/bin/env python3
# P1 Ingest (XLSX input): a group chat log stored as an XLSX workbook -> the same
# artifacts as the JSONL input (ingest.py).
#   - Columns are read by header name, never by position; rows without a turn number
#     or content are dropped; attachments become explicit placeholders.
#   - Member profile: the static header columns (name / role / traits) are imported
#     as-is (by=inherited:sheet, turn=0).
#   - The sheet's own label columns go to silver_labels and are not used as ground truth.
#   - Some exported XLSX files carry stylesheets that older openpyxl rejects, so
#     styles.xml and embedded images are stripped before reading; only cell values remain.
# Usage: python3 pipeline/ingest_xlsx.py <xlsx> --gid <name> [--sheet 0] [--profile]
#   output -> pipeline_runs/<gid>_<YYYYMMDD>/
# P1 接入(XLSX 输入): 以 xlsx 表格保存的群聊记录 → 与 JSONL 输入相同的产物。按表头名取列, 附件登记为占位符;
# 表头区成员信息直接导入档案, 表内标注列只作参考(silver_labels)。
import io, json, os, re, sys, time, zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.contract import ROOT
from pipeline.ingest import build_segments, build_profile
from pipeline.llm import code_commit

MIN_STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="1"><font/></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
              '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
              '<cellXfs count="1"><xf/></cellXfs></styleSheet>')
SILVER_COLS = ('是否问bot', '是否需要记忆功能', '是否拆分up', '主考察点1', '主考察点2', '主考察点3',
               '能力（考点）', '难易程度')


def load_rows(path, sheet_idx=0):
    import openpyxl, warnings
    warnings.filterwarnings('ignore')
    buf = io.BytesIO()
    with zipfile.ZipFile(path) as zi, zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zo:
        for it in zi.infolist():
            if it.filename == 'xl/styles.xml':
                zo.writestr(it.filename, MIN_STYLES)
            elif it.filename.startswith('xl/media/'):
                continue
            else:
                zo.writestr(it, zi.read(it.filename))
    buf.seek(0)
    wb = openpyxl.load_workbook(buf, read_only=True, data_only=True)
    ws = wb.worksheets[sheet_idx]
    return ws.title, [list(r) for r in ws.iter_rows(values_only=True)]


def _iso(v):
    if v is None:
        return None
    if hasattr(v, 'strftime'):
        return v.strftime('%Y-%m-%dT%H:%M:%S')
    s = str(v).strip().replace('/', '-')
    m = re.match(r'(\d{4}-\d{1,2}-\d{1,2})[ T](\d{1,2}:\d{2})(?::(\d{2}))?', s)
    if not m:
        return None
    d, hm, sec = m.groups()
    y, mo, da = d.split('-')
    return f'{y}-{int(mo):02d}-{int(da):02d}T{hm.zfill(5)}:{sec or "00"}'


def convert(path, gid, sheet_idx=0):
    title, rows = load_rows(path, sheet_idx)
    hdr_i = next(i for i, r in enumerate(rows[:30]) if any(c and '轮次' in str(c) for c in r))
    hdr = [str(c).strip() if c is not None else '' for c in rows[hdr_i]]
    col = {h: i for i, h in enumerate(hdr) if h}
    need = ['轮次', '聊天记录-发言人', '聊天记录-内容']
    miss = [k for k in need if k not in col]
    if miss:
        raise SystemExit(f'表头缺列 {miss}; 现有: {hdr[:20]}')
    g = lambda r, k: (r[col[k]] if k in col and col[k] < len(r) else None)

    # Group metadata and members (static header columns in the first rows)
    # 群元信息与成员(表头区静态列)
    group = {k: None for k in ('群背景', '群人数', '场景描述')}
    members = []
    for r in rows[hdr_i + 1:hdr_i + 60]:
        for k in list(group):
            kk = next((h for h in col if h.startswith(k)), None)
            if kk and g(r, kk) and not group[k]:
                group[k] = str(g(r, kk)).strip()
        name = g(r, '群成员名字')
        if name:
            members.append({'gid': gid, 'member': str(name).strip(),
                            'role': str(g(r, '群成员身份') or '').strip(),
                            'trait': str(g(r, '群成员特点') or '').strip(),
                            'by': 'inherited:sheet', 'turn': 0})

    msgs, silver, dropped = [], [], 0
    for ri, r in enumerate(rows[hdr_i + 1:], start=hdr_i + 2):
        t, who, text = g(r, '轮次'), g(r, '聊天记录-发言人'), g(r, '聊天记录-内容')
        try:
            t = int(t)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if who is None:
            dropped += 1
            continue
        who, text = str(who).strip(), (str(text).strip() if text is not None else '')
        fname, fdesc = g(r, '聊天记录-文件/图片名称'), g(r, '聊天记录-文件简单说明')
        att = []
        if fname:
            att.append({'type': 'image', 'name': str(fname).strip(), 'caption': str(fdesc or '').strip()})
            if not text:
                text = f'[图片: {str(fname).strip()} | {str(fdesc or "").strip()}]'  # attachment placeholder
        if not text and not att:
            dropped += 1
            continue
        msgs.append({'gid': gid, 'turn': t, 'who': who, 'text': text, 'ts': _iso(g(r, '时间')),
                     'kind': 'bot' if 'bot' in who else 'text',
                     'reply_to': g(r, 'reply_to'), 'attachments': att,
                     'src': {'track': 'xlsx', 'file': os.path.basename(path), 'sheet': title, 'row': ri}})
        labels = {k: str(g(r, k)).strip() for k in SILVER_COLS if k in col and g(r, k) not in (None, '')}
        if labels:
            silver.append({'gid': gid, 'turn': t, 'labels': labels, 'by': 'inherited:sheet',
                           'note': 'reference label from the sheet, not used as ground truth'})
    msgs.sort(key=lambda m: m['turn'])
    return group, members, msgs, silver, dropped


def main():
    args = sys.argv[1:]
    path = next((a for a in args if a.endswith('.xlsx')), None)
    if not path:
        raise SystemExit('用法: python3 pipeline/ingest_xlsx.py <xlsx> --gid <短名> [--sheet 0] [--profile]')
    gid = args[args.index('--gid') + 1] if '--gid' in args else os.path.basename(path)[:4]
    sheet = int(args[args.index('--sheet') + 1]) if '--sheet' in args else 0
    group, members, msgs, silver, dropped = convert(path, gid, sheet)
    # PIPE_RUN_DIR sets the run directory name; the default is <gid>_<YYYYMMDD>.
    # PIPE_RUN_DIR 指定 run 目录名, 默认 <gid>_<日期>。
    out = os.environ.get('PIPE_RUN_DIR') or f'{gid}_{time.strftime("%Y%m%d")}'
    out = out if os.path.isabs(out) else os.path.join(ROOT, 'pipeline_runs', out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'messages.jsonl'), 'w', encoding='utf-8') as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    with open(os.path.join(out, 'silver_labels.jsonl'), 'w', encoding='utf-8') as f:
        for s in silver:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    json.dump({'group': group, 'members': members}, open(os.path.join(out, 'members.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    segs = build_segments(msgs)
    json.dump(segs, open(os.path.join(out, 'segments.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    # Profile: write the static member entries first; with --profile, model-extracted
    # slots are merged in.
    # 档案: 先写入成员静态条目; 加 --profile 时再合并模型抽取的槽位。
    prof = [{'member': m['member'], 'slot': '人设', 'fact': f"{m['role']}；{m['trait']}".strip('；'),
             'turn': 0, 'quote': None, 'by': 'inherited:sheet', 'verified': 'inherited'} for m in members]
    meta = {'gid': gid, 'stage': 'P1(xlsx)', 'source': os.path.basename(path), 'sheet': sheet,
            'n_messages': len(msgs), 'n_bot': sum(1 for m in msgs if m['kind'] == 'bot'),
            'n_attachments': sum(1 for m in msgs if m['attachments']), 'n_dropped_rows': dropped,
            'n_segments': len(segs), 'n_members': len(members), 'n_silver': len(silver),
            'code_commit': code_commit(), 'run_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    if '--profile' in args:
        ok, rej = build_profile(msgs, out)
        prof += ok
        meta.update(profile_model_entries=len(ok), profile_rejected=len(rej))
    json.dump(prof, open(os.path.join(out, 'profile.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(meta, open(os.path.join(out, 'run_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f'→ {out}/')


if __name__ == '__main__':
    main()
