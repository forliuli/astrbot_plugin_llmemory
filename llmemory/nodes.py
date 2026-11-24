"""节点库。

一句话落成一颗节点：事件、强度、时刻、当时那句反读。
`raw`、`who` 是可留空的两格——上层愿意给就给，不给就是空串；
六维层本来也只靠自己这几样往下走。

边是时间上挨着：同一个窗口里落的节点互相连上，走得多的那条边就粗。
神游在这张图上走，走出来的东西归 muse，不在这儿开口。

落盘是追加写：谁变了就写谁，一行一颗，同一颗后写的盖前写的。
裁剪或行数太脏时才整本重写一遍。
"""

from __future__ import annotations

import heapq
import json
import math
import os
import random
import time
import uuid

from . import paths as P


def default_path() -> str:
    """数据目录里的节点册。一行一颗，名字照旧。"""
    return P.under("nodes.json")

EDGE_WINDOW = 1800.0     # 半小时内落的节点算挨着
MAX_EDGE = 8.0           # 一条边最多这么粗，再走也不更近
HALF_LIFE = 3 * 86400.0  # 热点按三天半衰算，一周前的还留一点余地

# 上限按库的大小分级：库还小的时候不用留太宽，长起来再往上放。
# 到顶两万颗，再往上裁到九成半，免得每落一颗都裁一遍。
CAP_TIERS = ((12000, 20000), (8000, 12000), (4000, 8000), (1500, 4000),
             (0, 1500))
MAX_NODES = 20000


def cap_for(count) -> int:
    """这个规模的库，最多容得下多少颗。MAX_NODES 是硬顶，分档在它下面走。"""
    try:
        count = max(0, int(count))
    except (TypeError, ValueError):
        count = 0
    tier = CAP_TIERS[-1][1]
    for floor, cap in CAP_TIERS:
        if count >= floor:
            tier = cap
            break
    try:
        hard = max(1, int(MAX_NODES))
    except (TypeError, ValueError):
        hard = 20000
    return min(tier, hard)


# ---- 收值的小夹子 ------------------------------------------------------

def _scale(v) -> float:
    """事件强度。坏值退回 1.0，越界的夹回 ±3。"""
    try:
        return max(-3.0, min(3.0, float(v)))
    except (TypeError, ValueError):
        return 1.0


def _ts(v) -> float:
    try:
        t = float(v)
    except (TypeError, ValueError):
        return 0.0
    return t if t > 0 else 0.0


def _weight(v) -> float:
    try:
        w = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(MAX_EDGE, w))


def _text(v) -> str:
    """raw、who 这类留白格：坏值一律当空串，不缺键也不崩。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return str(v)
    except Exception:
        return ""


def mood_phrases(line: str) -> set:
    """把反读拆成能对上的小块。两边都拆，比的是词不是整句。"""
    if not line:
        return set()
    text = str(line or "").strip()
    # 存的带「心情：」，当下的是 describe() 的裸句。两边都剥到只剩词块，
    # 否则「心情：有劲」和「有劲」永远对不上，热点的加分和暗处的挑句全哑。
    changed = True
    while changed:
        changed = False
        for head in ("心情：", "现在"):
            if text.startswith(head):
                text = text[len(head):].lstrip()
                changed = True
    text = text.replace("现在", "").replace("，", ",")
    return {p.strip(" 　。.") for p in text.split(",") if p.strip(" 　。.")}


# ---- 一颗节点 ----------------------------------------------------------

class Node:
    """一颗节点。全是情绪层自己的轨迹。"""

    __slots__ = ("id", "event", "scale", "ts", "mood", "raw", "who", "edges")

    def __init__(self, event: str = "", scale: float = 1.0, mood: str = "",
                 ts: float | None = None, nid: str = "", edges: dict | None = None,
                 raw: str = "", who: str = ""):
        self.id = nid or uuid.uuid4().hex[:10]
        self.event = str(event or "")
        self.scale = _scale(scale)
        self.ts = time.time() if ts is None else _ts(ts)
        self.mood = str(mood or "")
        self.raw = _text(raw)
        self.who = _text(who)
        self.edges = dict(edges or {})

    def as_dict(self) -> dict:
        return {"id": self.id, "event": self.event, "scale": self.scale,
                "ts": self.ts, "mood": self.mood, "raw": self.raw,
                "who": self.who, "edges": dict(self.edges)}

    @classmethod
    def from_dict(cls, data):
        """坏形状给 None，调用方跳过这一颗，不为它崩。"""
        if not isinstance(data, dict):
            return None
        nid = str(data.get("id") or "").strip()
        if not nid:
            return None
        edges = {}
        raw_edges = data.get("edges")
        if isinstance(raw_edges, dict):
            for key, val in raw_edges.items():
                key = str(key or "").strip()
                if not key or key == nid:
                    continue
                w = _weight(val)
                if w > 0:
                    edges[key] = w
        return cls(event=data.get("event"), scale=data.get("scale"),
                   mood=data.get("mood"), ts=data.get("ts"),
                   nid=nid, edges=edges,
                   raw=_text(data.get("raw", "")),
                   who=_text(data.get("who", "")))

    def __repr__(self) -> str:
        return f"<Node {self.id} {self.event} {self.scale:+.2f}>"


# ---- 节点册 ------------------------------------------------------------

class NodeBook:
    """一本节点册，落一个 JSONL 文件。读坏了当空的，不抛。"""

    __slots__ = ("path", "nodes", "_lines")

    def __init__(self, path: str = ""):
        self.path = path or default_path()
        self.nodes = {}
        self._lines = 0

    def __len__(self) -> int:
        return len(self.nodes)

    # ---- 读 ------------------------------------------------------------

    @staticmethod
    def _parse(text: str):
        """先按整份 JSON 认，认不出就一行一颗地认。给 (行列表, 是不是旧格式)。"""
        text = (text or "").strip()
        if not text:
            return [], False
        whole = None
        try:
            whole = json.loads(text)
        except (ValueError, TypeError):
            whole = None
        if isinstance(whole, dict) and "nodes" in whole:
            raw = whole["nodes"]
            if isinstance(raw, dict):
                return list(raw.values()), True
            if isinstance(raw, list):
                return raw, True
            return [], True
        if isinstance(whole, list):
            return whole, True
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(item, dict):
                rows.append(item)
        return rows, False

    def load(self) -> bool:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            self.nodes = {}
            self._lines = 0
            return False
        rows, legacy = self._parse(text)
        nodes = {}
        for item in rows:
            node = Node.from_dict(item)
            if node is not None:
                nodes[node.id] = node   # 后写的盖前写的
        self.nodes = nodes
        self._lines = len(rows)
        self._trim(compact=False)
        # 追加写会把被连粗的旧节点也各重写一行，行数虚高就整本重写一遍抹平
        if legacy or self._lines > len(nodes) * 1.5:
            self.save()                 # 旧格式顺手转成 JSONL
        return True

    # ---- 写 ------------------------------------------------------------

    def save(self) -> bool:
        """整本重写：一行一颗。裁剪、迁移、行数太脏时走这条。"""
        try:
            folder = os.path.dirname(self.path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for node in sorted(self.nodes.values(), key=lambda n: n.ts):
                    f.write(json.dumps(node.as_dict(), ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
            self._lines = len(self.nodes)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def _append(self, batch) -> bool:
        """追加写：变了的几颗各占一行。写不进就算了，内存里那份还在。"""
        batch = [n for n in batch if n is not None]
        if not batch:
            return True
        try:
            folder = os.path.dirname(self.path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for node in batch:
                    f.write(json.dumps(node.as_dict(), ensure_ascii=False) + "\n")
            self._lines += len(batch)
        except (OSError, TypeError, ValueError):
            return False
        # 一行一颗的账，脏到四倍以上就收一次。
        if self._lines > len(self.nodes) * 4 + 200:
            self.save()
        return True

    # ---- 落一颗 --------------------------------------------------------

    def add(self, event: str, scale: float = 1.0, mood: str = "",
            ts: float | None = None, window: float = EDGE_WINDOW,
            raw: str = "", who: str = "") -> Node:
        """落一颗，顺手跟窗口里刚落的那几颗连上，连过的也一起落盘。"""
        node = Node(event=event, scale=scale, mood=mood, ts=ts,
                    raw=raw, who=who)
        touched = self._link_recent(node, window)
        self.nodes[node.id] = node
        self._append([node] + touched)
        self._trim()
        return node

    def _link_recent(self, node: Node, window: float) -> list:
        """把新节点跟窗口里的连上，返回被连粗了的那些颗。"""
        if window <= 0:
            return []
        touched = []
        for other in self.nodes.values():
            if other.id == node.id:
                continue
            gap = abs(node.ts - other.ts)
            if gap > window:
                continue
            # 越近越结实，最远也有半格。
            w = 0.5 + 0.5 * (1.0 - gap / window)
            before = len(other.edges)
            self._link(node, other, w)
            if len(other.edges) != before or other.edges.get(node.id, 0) > 0:
                touched.append(other)
        return touched

    def _link(self, a: Node, b: Node, w: float) -> None:
        w = _weight(w)
        if w <= 0 or a.id == b.id:
            return
        a.edges[b.id] = _weight(a.edges.get(b.id, 0.0) + w)
        b.edges[a.id] = _weight(b.edges.get(a.id, 0.0) + w)

    def _trim(self, compact: bool = True) -> bool:
        """按分级上限裁老节点，顺手掐掉指向已走那几颗的边。

        过顶只多留百分之五的余量，够松一会儿再裁，不必落一颗裁一遍。
        """
        if not self.nodes:
            return False
        changed = False
        cap = cap_for(len(self.nodes))
        if len(self.nodes) > cap * 1.05:
            keep = sorted(self.nodes.values(), key=lambda n: n.ts,
                          reverse=True)[:cap]
            self.nodes = {n.id: n for n in keep}
            changed = True
        live = set(self.nodes)
        for node in self.nodes.values():
            if not node.edges:
                continue
            kept = {k: v for k, v in node.edges.items()
                    if k in live and v > 0}
            if len(kept) != len(node.edges):
                node.edges = kept
                changed = True
        if changed and compact:
            self.save()
        return changed

    # ---- 热点 ----------------------------------------------------------

    def score(self, node: Node, now_ts: float | None = None, mood: str = "") -> float:
        """一颗节点此刻有多容易浮上来：新、重、连着得多、跟当下对得上。"""
        now_ts = time.time() if now_ts is None else float(now_ts)
        gap = now_ts - node.ts
        recency = math.exp(-gap / HALF_LIFE) if gap > 0 else 1.0
        weight = 1.0 + min(abs(node.scale), 3.0) * 0.5
        degree = 1.0 + min(sum(node.edges.values()), MAX_EDGE * 2) * 0.06
        bonus = 1.0
        if mood:
            have, want = mood_phrases(node.mood), mood_phrases(mood)
            if have & want:
                bonus = 1.35
        return recency * weight * degree * bonus

    def hot(self, mood: str = "", now_ts: float | None = None,
            limit: int = 5, pool: int = 0) -> list:
        """眼下最容易浮上来的几颗。pool>0 时只在最近这些颗里挑，免得全是老事。"""
        if not self.nodes:
            return []
        items = list(self.nodes.values())
        if pool and len(items) > pool:
            items = sorted(items, key=lambda n: n.ts, reverse=True)[:int(pool)]
        now_ts = time.time() if now_ts is None else float(now_ts)
        limit = max(1, int(limit))
        if len(items) <= limit:
            return sorted(items, key=lambda n: n.ts, reverse=True)
        return heapq.nlargest(limit, items,
                              key=lambda n: (self.score(n, now_ts, mood), n.ts))

    def closest(self, mood: str, now_ts: float | None = None, window: float = 86400.0) -> Node | None:
        """窗口里跟当下心情最贴的一颗，没有就 None。"""
        now_ts = time.time() if now_ts is None else float(now_ts)
        want = mood_phrases(mood)
        if not want:
            return None
        best, best_hit = None, 0
        for node in self.nodes.values():
            if now_ts - node.ts > window:
                continue
            hit = len(want & mood_phrases(node.mood))
            if hit > best_hit:
                best, best_hit = node, hit
        return best

    # ---- 走 ------------------------------------------------------------

    def walk(self, start, steps: int = 2, rng=None) -> list:
        """从一颗节点往外走几跳。边上按粗细抽，走完给路上这几颗。"""
        rng = rng or random
        node = self.nodes.get(start) if isinstance(start, str) else start
        path = []
        for _ in range(max(0, int(steps))):
            if node is None:
                break
            path.append(node)
            neighbours = [(k, w) for k, w in node.edges.items() if k in self.nodes]
            if not neighbours:
                break
            total = sum(w for _, w in neighbours)
            if total <= 0:
                break
            pick = rng.uniform(0.0, total)
            acc, chosen = 0.0, neighbours[-1][0]
            for key, w in neighbours:
                acc += w
                if pick <= acc:
                    chosen = key
                    break
            node = self.nodes.get(chosen)
        if node is not None and node not in path:
            path.append(node)
        return path


def load_book(path: str = "") -> NodeBook:
    """一条路：给路径，读得出来就是旧的，读不出来就是新的一本。"""
    book = NodeBook(path)
    book.load()
    return book
