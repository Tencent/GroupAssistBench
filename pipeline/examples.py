#!/usr/bin/env python3
"""Few-shot examples injected into prompts: negatives first, then positives.

Four rules:
  1. Negatives come first: hand-written typical failures, each with a written
     "what is wrong", plus real failures caught by the gates.
  2. Examples are drawn across groups: every accessor takes exclude_gid, so a
     group never sees its own messages as examples.
  3. Examples are bucketed by stage (anchor / rt2_query / query_specific / is_request).
  4. Each example is rendered with its reason ("what is wrong" / "why it passes").

The examples are constants rather than files read at run time, so this package
sends exactly the prompts used to build the released benchmark. The `gid` field
is used only by rule 2 and never appears in a prompt.

提示词中的少样本示例: 先反例后正例; 跨群取样, 不用本群作样例; 按阶段分桶; 每条附判定理由。
示例固化为常量, 保证与构建发布数据集时的 prompt 一致; gid 只用于排除本群, 不进入 prompt。
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INBOX = os.path.join(ROOT, 'prompts', 'negative_inbox.jsonl')

# ---- Hand-written typical negatives, each with a written diagnosis ----
# 手工编写的典型反例, 每条附「错在哪」
CANON_NEG = [
    {'stage': 'anchor', 'gid': '电梯', 'case': '题问「低层是否交运行费」, 锚点挂到 #130——而 #130 实际是无障碍优先按钮的讨论, 与题无关(正确应为 #135/#38)', 'why': 'turn 指错整条作废; 挂锚前必须回读所指 turn 原文核对'},
    {'stage': 'anchor', 'gid': '1197', 'case': '题问「他为啥在屁股上装双闪灯」, 锚点挂到 bot 介绍某位人物的三条回复(#5/#6/#7)', 'why': 'bot 生成的知识回答不算群记忆; 锚点应指向成员产生的内容'},
    {'stage': 'anchor', 'gid': '1197', 'case': '题问「说说养蚕的事」, 锚点挂到提问消息自身(「说说养蚕的事呗, 听说它超废的」)', 'why': '锚点=提问本身, 零信息量; 锚点必须是提问之外、回答所依赖的历史事实'},
    {'stage': 'rt2_query', 'gid': '电梯', 'case': '题面写「大家为分工吵起来了」——时间线里根本没有分工争执, 前提是模型凭空编的', 'why': '虚构前提(Q8 硬拦); 题面的每个事实性陈述都必须能指认到具体 turn'},
    {'stage': 'rt2_query', 'gid': '2810', 'case': '模糊化删「60分」时把唯一定位信息一并删了——群内有多张地理卷, 模型定位到 50分/38分那张', 'why': '删泄漏时必须保住唯一定位; 题面仍须唯一指向目标记忆'},
    {'stage': 'rt2_query', 'gid': '投资', 'case': '引导语写「请从情绪和行为两方面回答」——无记忆模型凭「情绪」二字猜中「焦虑」, 得分 0→0.50', 'why': '维度名=GT 原子的主题词, 给出即泄漏; 引导只给数量与形式(「请分两点说明」)'},
    {'stage': 'rt2_query', 'gid': '2810', 'case': '题面问「怎么折算的」(方法), GT 却是「折算到50分制」(结果)——三档全 0', 'why': '问法与 GT 不匹配; 问方法就给方法型 GT, 问结果就给结果型 GT'},
    {'stage': 'query_specific', 'gid': '2810', 'case': '「重蹈 #1667 翻车: 漏算一张卷子导致完成量被低估」', 'why': '事后视角——模型与当年的元宝面对同一份信息, 用事后诸葛亮的标准考当时的人(Q2c 硬拦)'},
    {'stage': 'is_request', 'gid': '2810', 'case': '「@元宝 记住我们都是干啥的哈」', 'why': '写入设定不是请求; 它建立记忆而非消费记忆'},
]

# ---- Real requests phrased irregularly (is_request positives) ----
# 非规范表达的真实请求(is_request 正例)
CANON_POS_ISREQ = [
    {'gid': '1197', 'case': '「？？红红？！」(成员对 bot 回复的困惑追问)', 'why': '情绪化追问=要求 bot 解释, 是请求'},
    {'gid': '1197', 'case': '「@元宝」(单独出现)', 'why': '@ 即召唤服务, 是请求'},
    {'gid': '1197', 'case': '「@元宝 发广告没商量，直接清走！」', 'why': '命令式=要 bot 执行群规, 是请求'},
    {'gid': '1197', 'case': '「我唱的是大张伟的《永远唱不完的歌》，你是不是没看链接内容？」', 'why': '引用+质疑=要求 bot 重新处理, 是请求'},
]

# ---- Human-labelled anchors from other groups: one per anchor type, concrete values ----
# 跨群人工标注的锚点正例: 五种类型各一条, fact 为具体值
ANCHOR_POS = [
    {'gid': '1197', 'case': '[事实] #53 刚讲过eat/ate的时态', 'why': 'fact 写具体值、turn 指向确立点——事实类锚点的标准形态'},
    {'gid': '1197', 'case': '[人设] #3 元宝爱玩笑型人设(高频表情)', 'why': 'fact 写具体值、turn 指向确立点——人设类锚点的标准形态'},
    {'gid': '1197', 'case': "[偏好] #190 群内'理财别讲代码'的口味(user_2曾提,user_1此处沿用)", 'why': 'fact 写具体值、turn 指向确立点——偏好类锚点的标准形态'},
    {'gid': '2810', 'case': '[约束] #9 约束:累了就别死磕、放过', 'why': 'fact 写具体值、turn 指向确立点——约束类锚点的标准形态'},
    {'gid': '2810', 'case': '[承诺] #46 user_1(同学B)定了6:30闹钟', 'why': 'fact 写具体值、turn 指向确立点——承诺类锚点的标准形态'},
]

# ---- Meta-descriptions caught by the gates (query_specific negatives) ----
# 闸门拦下的元描述, 作 query_specific 反例
GATE_NEG = [
    {'gid': '', 'case': '「回答必须明确提及倒计时天数为21天，截止日期为6月14日生地会考，若仅泛称倒计时或考」', 'why': '元描述——「必须明确提及/指出」是给判分者的指令, 不是对回答的要求(Q1 硬拦)'},
    {'gid': '', 'case': '「回答必须明确提及#1030中user_1承诺要鞭策且恢复督促力度，泛称鞭策无具体承诺」', 'why': '元描述——「必须明确提及/指出」是给判分者的指令, 不是对回答的要求(Q1 硬拦)'},
    {'gid': '', 'case': '「回答必须明确提及群内#986消息中6月14日生地会考倒计时21天的具体信息，且体现尊」', 'why': '元描述——「必须明确提及/指出」是给判分者的指令, 不是对回答的要求(Q1 硬拦)'},
    {'gid': '', 'case': '「回答中明确提及6月14日生地会考倒计时21天（#986），并结合此信息给出合理作业复」', 'why': '元描述——「必须明确提及/指出」是给判分者的指令, 不是对回答的要求(Q1 硬拦)'},
]

# ---- query_specific criteria from a human-reviewed rubric (positives) ----
# 人工审定评分标准中的 query_specific 条目, 作正例
QS_POS = [
    {'gid': '投资', 'case': '「对症具体：≥2条可执行且能说出「为什么适合她」的措施(定投自动化/APP降频/sleep test)」', 'why': '落到可执行行为且带个性化钩子(「为什么适合她」), 过双重测试'},
    {'gid': '投资', 'case': '「会计杠杆：用数字敏感特质做量化安抚(波动区间/回本测算)」', 'why': '落到可执行行为且带个性化钩子(「为什么适合她」), 过双重测试'},
    {'gid': '投资', 'case': '「延续既有方案：呼应元宝先前#296「少看为妙」节奏/#605急救三步法，不另起炉灶」', 'why': '落到可执行行为且带个性化钩子(「为什么适合她」), 过双重测试'},
    {'gid': '2810', 'case': '「裁决规则：当下显式请求 > 历史人设默认值」', 'why': '落到可执行行为且带个性化钩子(「为什么适合她」), 过双重测试'},
]


def _keep(e, exclude_gid):
    return not (exclude_gid and e['gid'] == str(exclude_gid))


def _load_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def gate_negatives(stage):
    return list(GATE_NEG) if stage == 'query_specific' else []


def anchor_positives(exclude_gid=None, k=5):
    return [e for e in ANCHOR_POS if _keep(e, exclude_gid)][:k]


def qs_positives(exclude_gid=None, k=2):
    return [e for e in QS_POS if _keep(e, exclude_gid)][:k]


def negatives_for(stage, exclude_gid=None, k=4):
    pool = [e for e in CANON_NEG if e['stage'] == stage and _keep(e, exclude_gid)]
    pool += gate_negatives(stage)
    return pool[:k]


def positives_for(stage, exclude_gid=None, k=3):
    if stage == 'anchor':
        return anchor_positives(exclude_gid, k=5)
    if stage == 'query_specific':
        return qs_positives(exclude_gid, k=2)
    if stage == 'is_request':
        return [e for e in CANON_POS_ISREQ if _keep(e, exclude_gid)][:k + 1]
    return []


def render(stage, exclude_gid=None):
    """Render a stage's negatives, then its positives; returns '' when there are none.
    渲染某阶段的反例与正例(反例在前); 没有则返回空串。
    """
    neg, pos = negatives_for(stage, exclude_gid), positives_for(stage, exclude_gid)
    parts = []
    if neg:
        parts.append('# 反例库(闸门拦下的真实错误, 不要犯同类错)\n' + '\n'.join(
            f'- {e["case"]}\n  错在哪: {e["why"]}' for e in neg))
    if pos:
        parts.append('# 正例(过闸产物)\n' + '\n'.join(
            f'- {e["case"]}\n  为什么过: {e["why"]}' for e in pos))
    return ('\n\n' + '\n\n'.join(parts)) if parts else ''


def append_failure(stage, gid, case, rule, detail=''):
    """Feedback loop: gate failures are written to an inbox (deduplicated). After a human
    adds "what is wrong", they can be promoted into CANON_NEG.
    反馈环: 闸门失败样本写入收件箱(去重), 人工补充「错在哪」后可并入 CANON_NEG。
    """
    row = {'stage': stage, 'gid': str(gid), 'case': str(case)[:120],
           'rule': rule, 'detail': str(detail)[:80], 'curated': False}
    seen = {(r['stage'], r['case']) for r in _load_jsonl(INBOX)}
    if (row['stage'], row['case']) in seen:
        return
    os.makedirs(os.path.dirname(INBOX), exist_ok=True)
    with open(INBOX, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    for st in ('anchor', 'rt2_query', 'query_specific', 'is_request'):
        blk = render(st, exclude_gid=sys.argv[1] if len(sys.argv) > 1 else None)
        print(f'===== {st} ====={blk}\n')
