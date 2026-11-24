"""神游。

没有要紧事的时候，从热点往外走几跳，把路上那几颗连成一句碎片。
碎片只落盘，不当场说——想说的那一层归自唤醒，它自己挑时候。

一趟只走一步，带出来的那颗就是这一分钟的碎片。
"""

from __future__ import annotations

import json
import os
import random
import time

from .nodes import NodeBook, mood_phrases, mood_hit

DEFAULT_PATH = os.path.join("plugin_data", "暗处", "muse.jsonl")

HOPS = 1                   # 一次扩散只带一颗：从起点沿边走一步，那一颗就是带出来的
NIGHT_START, NIGHT_END = 23, 6
MAX_LINES = 2000           # 碎片不无限堆，超了留下后半段

_EVENT_CN = {
    "praise": "被夸那下",
    "criticism": "被挑那下",
    "bonding": "凑近那下",
    "stress": "压过来那下",
    "relaxation": "松下来那下",
    "talk": "说着话那会儿",
}


def _now() -> float:
    return time.time()


def _mood_core(line: str) -> str:
    """反读去掉「现在」和句末标点，剩下的才接得上话。"""
    text = str(line or "").strip()
    # 头可能叠着：老节点存成「心情：心情：现在…」，只剥一层会剩一半。
    changed = True
    while changed:
        changed = False
        for head in ("心情：", "现在"):
            if text.startswith(head):
                text = text[len(head):].lstrip()
                changed = True
    return text.strip(" 　。.，,")


def is_night(now_ts: float | None = None) -> bool:
    hour = time.localtime(_now() if now_ts is None else now_ts).tm_hour
    return hour >= NIGHT_START or hour < NIGHT_END


def hops_for(now_ts: float | None = None) -> int:
    """一趟走几步。一次扩散只走一步，带出一颗。"""
    return HOPS


def compose(path: list) -> str:
    """把路上这几颗拼成一句。只写轨迹，不替谁说话。"""
    if not path:
        return ""
    head = _EVENT_CN.get(path[0].event, "")
    first = _mood_core(path[0].mood)
    if not head:
        return first
    if len(path) == 1:
        return f"{head}，{first}。" if first else f"{head}。"
    last = _mood_core(path[-1].mood)
    if last and last != first:
        return f"{head}，{first}；走到这儿，{last}。"
    # 路上两头的反读一样：说中间那句，别拿同一句凑两次。
    middles = [m for m in (_mood_core(n.mood) for n in path[1:-1]) if m and m != first]
    if middles:
        return f"{head}，{first}；中间拐到{middles[0]}。"
    return f"{head}，{first}。" if first else f"{head}。"


# ---- 挑起点 ------------------------------------------------------------

def pick_start(book: NodeBook, mood: str = "", now_ts: float | None = None,
               avoid: set | None = None, rng=None) -> object:
    """从哪儿开始走。情绪偏就先认那股情绪，没偏就取近来的。"""
    now_ts = _now() if now_ts is None else now_ts
    rng = rng or random
    pool = []
    want = mood_phrases(mood) if mood else set()
    if want:
        # 情绪偏的时候要的是「这股情绪最重的那颗」，不能先被最近几十颗切掉
        hits = [n for n in book.nodes.values() if mood_hit(mood_phrases(n.mood), want)]
        if hits:
            hits.sort(key=lambda n: (book.score(n, now_ts, mood), n.ts), reverse=True)
            pool = hits[:6]
    if not pool:
        pool = book.hot(mood=mood, now_ts=now_ts, limit=6, pool=40)
    if not pool:
        return None
    avoid = avoid or set()
    fresh = [n for n in pool if n.id not in avoid]
    return rng.choice(fresh or pool)


def wander(book: NodeBook, mood: str = "", now_ts: float | None = None,
           hops: int | None = None, avoid: set | None = None, rng=None) -> dict | None:
    """一次扩散：挑一颗起点，沿边抽一步，带出一颗。走不动给 None，不硬编。"""
    now_ts = _now() if now_ts is None else now_ts
    rng = rng or random
    start = pick_start(book, mood, now_ts, avoid, rng)
    if start is None:
        return None
    steps = hops_for(now_ts) if hops is None else max(1, int(hops))
    path = book.walk(start, steps=steps, rng=rng)
    if not path:
        return None
    for a, b in zip(path, path[1:]):
        book.hebb(a, b, abs(getattr(b, "scale", 0.0)))
    return {"ts": now_ts,
            "line": compose(path),
            "ids": [n.id for n in path],
            "events": [n.event for n in path],
            "start": path[0].id,
            "end": path[-1].id,
            "night": is_night(now_ts),
            "hops": len(path)}


# ---- 落盘 --------------------------------------------------------------

def remember(path: str, item: dict) -> bool:
    """一行一条，追加。写不进就算了，一次神游不值得报错。"""
    if not isinstance(item, dict) or not item.get("line"):
        return False
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        _rotate(path)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _rotate(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, ValueError):
        return
    if len(lines) <= MAX_LINES:
        return
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines[-MAX_LINES // 2:])
        os.replace(tmp, path)
    except OSError:
        pass


def fragments(path: str, limit: int = 20) -> list:
    """读回后几条。文件不在就是空的，不当错误。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, ValueError):
        return []
    out = []
    for line in lines[-max(1, int(limit)):]:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("line"):
            out.append(item)
    return out


def fresh(path: str, since_ts: float, limit: int = 20) -> list:
    """这段时间里新落的碎片。自唤醒就拿它当话头。"""
    out = []
    for item in fragments(path, limit=MAX_LINES):
        try:
            if float(item.get("ts") or 0) >= since_ts:
                out.append(item)
        except (TypeError, ValueError):
            continue
    return out[-max(1, int(limit)):]


def avoid_set(path: str, n: int = 12) -> set:
    """最近走过的那几个起点，先绕开，别老在原地转。"""
    starts = set()
    for item in fragments(path, limit=n):
        key = item.get("start")
        if key:
            starts.add(str(key))
    return starts


def matches(item: dict, mood: str) -> bool:
    """这条碎片跟当下的心情对不对得上。对得上才值得拿来说。"""
    want = mood_phrases(mood)
    if not want or not isinstance(item, dict):
        return False
    return bool(want & mood_phrases(item.get("line", "")))


def run_once(book: NodeBook, path: str = "", mood: str = "",
             now_ts: float | None = None) -> dict | None:
    """一次完整的神游：走、拼、落。返回落下的那条，没走成给 None。"""
    path = path or DEFAULT_PATH
    item = wander(book, mood=mood, now_ts=now_ts, avoid=avoid_set(path))
    if item is None:
        return None
    remember(path, item)
    return item
