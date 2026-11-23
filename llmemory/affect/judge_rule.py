# -*- coding: utf-8 -*-
"""规则档判词：不调模型，纯查表。

一句话进来 → 切词 → 查正负表 → 处理否定与程度 → 压成 valence / arousal / tension
→ 映射成 neurostate 认的 (event_type, scale)。

和 parse_judgement 输出同一个形状，所以可以直接喂 apply_judgement。
判不出来就返回 None，跟模型档一样——不许猜。

已知它会错的地方：
- 反讽。「你可真行」在字面上就是夸，规则看不出来。
- 骂人和发愁分不开，靠句子里有没有「你」硬掰——有你是批评，没你是压力。

词表是内置的小表，够跑通就行。用户嫌哪个词判错，面板里加成对（词 + 分）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- 词表

# 分值 1=轻 2=中 3=重
# 「好」不进正表：它八成是程度词（好累 / 好讨厌 / 好喜欢），留在 DEGREE 里。
POS_WORDS = {
    "棒": 2, "厉害": 2, "可爱": 2, "喜欢": 2, "爱": 2, "谢谢": 2,
    "感谢": 2, "辛苦": 1, "赞": 2, "优秀": 2, "漂亮": 2, "好看": 2, "好听": 1,
    "舒服": 2, "开心": 2, "高兴": 2, "满意": 2, "放心": 2, "靠谱": 2, "牛": 2,
    "聪明": 2, "温暖": 2, "治愈": 2, "安心": 2, "踏实": 2, "温柔": 2, "贴心": 3,
    "用心": 2, "认真": 1, "佩服": 2, "可爱死": 3, "绝了": 2,
    "不错": 2, "挺好": 2, "真好": 2, "有趣": 1, "有意思": 1,
}

NEG_WORDS = {
    "烦": 2, "讨厌": 3, "难过": 2, "难受": 2, "累": 1, "烦人": 2, "无聊": 1,
    "气": 2, "生气": 3, "失望": 3, "担心": 2, "怕": 2, "焦虑": 3, "郁闷": 2,
    "伤心": 3, "崩溃": 3, "痛苦": 3, "糟糕": 3, "垃圾": 3, "烂": 2, "差": 2,
    "错": 2, "不行": 2, "恶心": 3, "吵": 1, "蠢": 2, "笨": 1, "傻": 1,
    "离谱": 3, "无语": 2, "头疼": 2, "烦死": 3, "受不了": 3, "算了": 1,
    "没意思": 2, "熬": 1, "撑不住": 3, "好累": 2, "想哭": 3, "委屈": 2,
}

# 黏人/亲近这一路单独走，命中就往 bonding 推
INTIMATE_WORDS = {
    "抱": 2, "抱抱": 2, "亲": 2, "亲亲": 2, "贴": 1, "想你": 3, "想见你": 3,
    "陪我": 2, "黏": 2, "蹭": 1, "摸": 1, "晚安": 1, "早安": 1, "乖": 1,
    "宝贝": 2, "在一起": 2, "抱一下": 2, "靠一会儿": 1,
}

# 外头压过来的事，往 stress 推
STRESS_WORDS = {
    "加班": 2, "赶": 1, "来不及": 2, "熬夜": 2, "失眠": 2, "吵架": 3, "催": 2,
    "考试": 2, "复习": 1, "答辩": 2, "面试": 2, "赶工": 2, "截稿": 2, "开会": 1,
}

DEGREE = {
    "特别": 1.8, "非常": 1.8, "超": 1.8, "巨": 1.9, "贼": 1.8, "太": 1.7,
    "这么": 1.6, "那么": 1.6, "那么那么": 1.8, "很": 1.6, "真": 1.5,
    "还挺": 1.5, "好": 1.4, "蛮": 1.4, "有点": 0.6, "稍微": 0.5, "一点": 0.5,
    "不太": 0.5,
}

NEGATORS = ("不", "没", "别", "勿", "无", "非", "莫")

# 程度词和否定词都不许跨标点：「你真好，谢谢」里的「好」不是「谢谢」的倍率。
BOUNDARY = "，。！？、；：,.!?;:\n\t "


def _local_head(text: str, pos: int, width: int) -> str:
    """取情感词前 width 个字，碰到标点就截断。"""
    head = text[max(0, pos - width):pos]
    cut = max((head.rfind(c) for c in BOUNDARY), default=-1)
    return head[cut + 1:]

# 第二人称。有它才判成「批评」，没有就是自己扛着 → 压力。
SECOND_PERSON = ("你", "您")

# ---------------------------------------------------------------- 阈值

VAL_PRAISE = 0.35   # 往上的高分
VAL_BONDING = 0.12  # 轻度的好
VAL_STRESS = -0.12  # 轻度的坏
VAL_CRITICISM = -0.35


@dataclass
class Verdict:
    event: str | None            # 五个事件之一，判不出为 None
    scale: float
    valence: float               # -1..1
    arousal: float               # 0..1
    tension: float               # 0..1
    trace: list[str] = field(default_factory=list)

    def as_tuple(self):
        return None if self.event is None else (self.event, self.scale)

    def explain(self) -> str:
        if not self.trace:
            return "没认出来，不动"
        tail = f" → {self.event}×{self.scale:.2f}" if self.event else " → 静默"
        return "；".join(self.trace) + tail


def _scan(text: str, table: dict) -> list[tuple[int, str, int, int]]:
    """最长匹配切词。返回 (位置, 词, 分, 长度)。"""
    if not text or not table:
        return []
    maxlen = max(len(w) for w in table)
    hits = []
    i, n = 0, len(text)
    while i < n:
        for L in range(min(maxlen, n - i), 0, -1):
            cand = text[i:i + L]
            if cand in table:
                hits.append((i, cand, table[cand], L))
                i += L
                break
        else:
            i += 1
    return hits


def _negated(text: str, pos: int) -> str:
    """词前面 2 个字里头有没有否定词。返回命中的那个，没有返回空串。"""
    head = _local_head(text, pos, 2)
    for neg in NEGATORS:
        if neg in head:
            return neg
    return ""


def _degree(text: str, pos: int) -> tuple[str, float]:
    """词前面 3 个字里、离词最近的程度词（不跨标点）。"""
    head = _local_head(text, pos, 3)
    best, val, best_end = "", 1.0, -1
    for word, mul in DEGREE.items():
        idx = head.rfind(word)
        if idx < 0:
            continue
        end = idx + len(word)
        if end > best_end:          # 越靠后越贴近情感词，优先
            best, val, best_end = word, mul, end
    return best, val


def judge(text: str) -> Verdict:
    """跑一遍。永远给得出 Verdict，判不出的时候 event=None。"""
    if not isinstance(text, str):
        return Verdict(None, 0.0, 0.0, 0.0, 0.0)
    text = text.strip()
    if not text:
        return Verdict(None, 0.0, 0.0, 0.0, 0.0)

    trace: list[str] = []
    total, pos, intimate, stress = 0.0, 0.0, 0.0, 0.0

    for bucket, table, sign in (
        ("好", POS_WORDS, 1.0),
        ("坏", NEG_WORDS, -1.0),
        ("黏", INTIMATE_WORDS, 1.0),
        ("压", STRESS_WORDS, -1.0),
    ):
        for pos, word, score, _len in _scan(text, table):
            value = score * sign
            neg = _negated(text, pos)
            if neg and bucket in ("好", "坏"):
                value = -value
                trace.append(f"{neg}…{word} 翻")
            elif neg:
                # 「别催」「不想你」是把那件事撤掉，翻成正向就离谱了
                trace.append(f"{neg}…{word} 撤")
                continue
            deg, mul = _degree(text, pos)
            if deg:
                value *= mul
                trace.append(f"{deg}×{mul:g} {word}")
            else:
                trace.append(f"{word}{value:+.0f}")
            total += value
            if bucket == "好":
                pos += abs(value)
            elif bucket == "黏":
                intimate += abs(value)
            elif bucket == "压":
                stress += abs(value)

    # 标点：感叹号抬唤醒，问号抬一点紧张
    bangs = text.count("!") + text.count("！") + text.count("～") * 0.5
    qs = text.count("?") + text.count("？")
    repeats = sum(1 for i in range(1, len(text)) if text[i] == text[i - 1])

    valence = total / (abs(total) + 2.5) if total else 0.0
    arousal = min(1.0, abs(total) / 5.0 + bangs * 0.15 + repeats * 0.1)
    tension = min(1.0, max(0.0, -valence) * 0.7 + min(qs, 2) * 0.15 + stress * 0.05)

    event, scale = _map(valence, pos, intimate, stress, arousal, text)
    return Verdict(event, scale, round(valence, 3), round(arousal, 3), round(tension, 3), trace)


def _map(valence: float, pos: float, intimate: float, stress: float,
         arousal: float, text: str) -> tuple[str | None, float]:
    """valence 定方向和力度，黏人掰向 bonding，压力掰向 stress。"""
    stress_ratio = stress / (abs(valence) * 2.5 + 1.0)   # 负面里有多少是外头压来的
    blame = any(p in text for p in SECOND_PERSON)

    if valence >= VAL_BONDING and intimate >= 1.0 and intimate >= pos and stress < 1.0:
        return "bonding", min(2.0, 0.8 + intimate * 0.15)
    if valence >= VAL_PRAISE:
        return "praise", min(2.0, 1.0 + (valence - VAL_PRAISE) * 2 + arousal * 0.3)
    if valence >= VAL_BONDING:
        return "bonding", min(2.0, 0.7 + valence)
    # 负面：骂我（句子里有你 + 不是在说自己累）走批评，其余都是压力
    if valence <= VAL_CRITICISM and blame and stress_ratio < 0.6:
        return "criticism", min(2.0, 1.0 + abs(valence + VAL_CRITICISM) * 2 + arousal * 0.3)
    if valence <= VAL_STRESS:
        return "stress", min(2.0, 0.8 + abs(valence) + arousal * 0.3)
    return None, 0.0


def judge_tuple(text: str) -> tuple[str, float] | None:
    """跟 parse_judgement 同形，直接喂 apply_judgement。"""
    return judge(text).as_tuple()


if __name__ == "__main__":
    for s in (
        "你今天怎么这么可爱",
        "你可真行",
        "我有点累了，先睡了",
        "抱抱我",
        "谢谢，辛苦你了",
        "我好讨厌这样，别这样了",
        "你真讨厌，怎么这么笨",
        "明天要考试了，烦死了",
        "嗯",
        "你真好，谢谢你陪我",
        "累死我了，不说了",
    ):
        v = judge(s)
        print(f"{s!r:24} val {v.valence:+.2f} ar {v.arousal:.2f} te {v.tension:.2f}  {v.explain()}")
