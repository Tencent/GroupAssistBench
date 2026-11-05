#!/usr/bin/env python3
# Data contract: canonical field names for instances, and a read-only adapter for
# externally supplied human annotations. Field names are defined only here.
#   annotations : load_annotations() translates field names on read; the source
#                 file is never modified.
#   gates       : produced by pipeline/gates.py as gates.jsonl.
# 数据契约: 题目的规范字段名, 以及人工标注的只读适配器。字段名只在此处定义。
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load(p):
    """Load JSONL; optional reference/gold files may be absent in the public package."""
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

REQUIRED = {'L1': ('iid', 'gid', 'difficulty', 'query', 'gt_atoms'),
            'L2': ('iid', 'gid', 'difficulty', 'query', 'criteria')}

def validate_item(it):
    errs = [f'missing:{f}' for f in REQUIRED[it['layer']] if not it.get(f)]
    for c in (it.get('criteria') or []):
        if not all(k in c for k in ('check', 'points', 'kind')):
            errs.append('criteria:missing_field')
            break
    return errs


# ---------- Human annotations: read-only adapter (renames fields, never edits the source) ----------
INTEGRATED = os.environ.get('PIPE_ANNOTATIONS_JSONL', 'data/annotations_integrated.jsonl')
# Provenance of each human label: the integrated file merges a first-pass annotation
# round with a later top-up round; both are recorded as anonymous annotator ids.
# integrated 文件合并了两轮人工标注(首轮 + 补标), 统一记为匿名标注者编号。
_BY = {'annotator_a': 'human:annotator_a', 'annotator_b': 'human:annotator_b'}

def load_annotations(root=ROOT, path=INTEGRATED, gid=None):
    out = []
    for r in _load(os.path.join(root, path)):
        g = str(r['group_id'])[:4]
        if gid and g != gid:
            continue
        by = _BY.get(r.get('source'), 'human:?')
        out.append({
            'aid': f'{g}#{r["idx"]}', 'gid': g, 'turn': int(r['idx']),
            'query': r.get('query'),
            'labels': {
                'grounding':  {'v': r.get('grounding'),  'by': by},
                'layer':      {'v': r.get('layer'),      'by': by},
                'difficulty': {'v': r.get('difficulty'), 'by': by},
                'is_request': {'v': bool(r.get('is_request')), 'by': by},
            },
            'anchors': [{'turn': a.get('idx'), 'fact': a.get('fact'), 'type': a.get('type'),
                         'by': by}
                        for a in (r.get('linked_memory') or []) if isinstance(a, dict)],
            '_adapter': 'integrated_v1',  # marks this as an adapter view, not a migrated file
        })
    return out
