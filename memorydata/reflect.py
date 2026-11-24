# -*- coding: utf-8 -*-
"""反思：把新落的几句话重新看一遍，只挑能并成一句的那些，落成一颗节点。

可填可不填——上层没配模型（model_url / model_name 任一为空）这件事就整个不做。
判不清、只是话题相关、只是那阵子情绪起伏，都不写；空着手回来也算个结果。

游标只记「看到哪一颗」，所以每批不重复看。
调用失败（没回话）游标不动，下一轮还看这一批；回了话但格式不对，这一批算看过，
不然同一批能在原地卡一辈子。
"""

from __future__ import annotations

import json
import os
import time

MIN_BATCH = 2        # 少于两句不值得看
MAX_BATCH = 8
MAX_PENDING = 40     # 积压太多只取最近的四十条
NEIGHBOR_K = 3       # 每句带几个旁边挨着的旧话
MIN_CHARS = 6
MAX_CHARS = 160
MIN_CONF = 0.6
EVENT = "反思"

_SYSTEM = "\n".join([
    "你在把这个人过往的几句话重新看一遍，只做一件事：挑出能并成一句的那些。",
    "算的只有两种：两句互相矛盾、后一句把前一句改了；或者几句合起来能定下一件事。",
    "只是话题相关、只是那阵子情绪起伏、判不清，都不算。",
    "只输出一行 JSON，不要解释：",
    '{"reflect": true, "text": "并成的那一句，不超过一百六十字", "confidence": 0.0}',
    '不算就输出 {"reflect": false}。confidence 是 0 到 1 之间的小数。',
])


def _ago(ts, now=None):
    now = time.time() if now is None else now
    d = max(0.0, now - float(ts or 0))
    if d < 3600:
        return "%d分钟前" % max(1, int(d // 60))
    if d < 86400:
        return "%d小时前" % int(d // 3600)
    return "%d天前" % int(d // 86400)


def _body(node) -> str:
    """给模型看的这一句：尽量用原话，没有就拿判出来的那句。"""
    text = (getattr(node, "raw", "") or "").strip() or (getattr(node, "event", "") or "").strip()
    return " ".join(text.split())[:160]


def _fmt(node, idx, now) -> str:
    who = (getattr(node, "who", "") or "").strip() or "谁"
    return "%d. %s，%s：%s" % (idx, _ago(getattr(node, "ts", 0), now), who, _body(node))


class Reflector:
    """游标 + 一批的挑法 + 三关。模型从外面递进来，自己不碰网络。"""

    def __init__(self, cursor_path: str = ""):
        self.cursor_path = cursor_path or ""
        self.cursor = self._load()

    # ---- 游标 ----------------------------------------------------------

    def _load(self) -> dict:
        if not self.cursor_path:
            return {}
        try:
            with open(self.cursor_path, encoding="utf-8") as f:
                got = json.load(f)
            return got if isinstance(got, dict) else {}
        except Exception:
            return {}                      # 头一回，或者文件坏了，就当没看过

    def _save(self, last_id: str) -> None:
        self.cursor = {"id": str(last_id or "")}
        if not self.cursor_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cursor_path), exist_ok=True)
            with open(self.cursor_path, "w", encoding="utf-8") as f:
                json.dump(self.cursor, f, ensure_ascii=False)
        except Exception:
            pass                           # 游标记不住就多跑一遍，不算坏事

    # ---- 挑 ------------------------------------------------------------

    def pending(self, book) -> list:
        """游标之后还没看过的几颗，按落地顺序。游标那颗被裁了就从头数最近那批。"""
        try:
            nodes = list(book.nodes.values())
        except Exception:
            return []
        if not nodes:
            return []
        last = str((self.cursor or {}).get("id") or "")
        start = 0
        if last:
            for i in range(len(nodes) - 1, -1, -1):
                if nodes[i].id == last:
                    start = i + 1
                    break
        ready = [n for n in nodes[start:] if _body(n)]
        if len(ready) > MAX_PENDING:
            ready = ready[-MAX_PENDING:]
        return ready

    def _neighbors(self, node, book, k: int = NEIGHBOR_K) -> list:
        """这颗旁边挨着的旧话，边粗的在前。"""
        edges = getattr(node, "edges", None) or {}
        out = []
        for nid in sorted(edges, key=lambda x: -float(edges.get(x) or 0)):
            nb = book.nodes.get(nid)
            if nb is not None and _body(nb):
                out.append(nb)
            if len(out) >= k:
                break
        return out

    def prompt(self, batch, book, now=None) -> list:
        now = time.time() if now is None else now
        lines = [_fmt(n, i + 1, now) for i, n in enumerate(batch)]
        body = "【新落的几句】\n" + "\n".join(lines)
        side, seen, idx = [], {n.id for n in batch}, 0
        for node in batch:
            for nb in self._neighbors(node, book):
                if nb.id in seen:
                    continue
                seen.add(nb.id)
                idx += 1
                side.append("- " + _ago(nb.ts, now) + "，" +
                            ((nb.who or "").strip() or "谁") + "：" + _body(nb))
                if idx >= NEIGHBOR_K * 2:
                    break
            if idx >= NEIGHBOR_K * 2:
                break
        if side:
            body += "\n\n【旁边挨着的旧话，不编号】\n" + "\n".join(side)
        return [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": body}]

    # ---- 读回话 --------------------------------------------------------

    def parse(self, text):
        """→ (真/假, 内容) ；读不出来给 (None, None)。"""
        text = (text or "").strip()
        if not text:
            return None, None
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j <= i:
            return None, None
        try:
            obj = json.loads(text[i:j + 1])
        except Exception:
            return None, None
        if not isinstance(obj, dict):
            return None, None
        if not obj.get("reflect"):
            return False, None
        body = " ".join(str(obj.get("text") or "").split()).strip()
        if not body:
            return False, None
        try:
            conf = float(obj.get("confidence"))
        except (TypeError, ValueError):
            conf = 0.0
        return True, {"text": body[:MAX_CHARS], "conf": max(0.0, min(1.0, conf))}

    # ---- 跑一批 --------------------------------------------------------

    def run(self, book, call, link=None, now=None) -> int:
        """看一批。落了返回 1，没落返回 0。call(messages) 回正文。"""
        batch = self.pending(book)
        if len(batch) < MIN_BATCH:
            return 0
        batch = batch[:MAX_BATCH]
        messages = self.prompt(batch, book, now)
        try:
            text = call(messages)
        except Exception:
            return 0                       # 没回话，游标不动，下一轮还看这一批
        if not str(text or "").strip():
            return 0
        last = batch[-1].id
        ok, got = self.parse(text)
        if ok is None or not ok or not got:
            self._save(last)               # 回话了，这一批算看过
            return 0
        body = got["text"]
        if not (MIN_CHARS <= len(body) <= MAX_CHARS) or got["conf"] < MIN_CONF:
            self._save(last)
            return 0
        try:
            known = set()
            for n in book.nodes.values():
                known.add((getattr(n, "event", "") or "").strip())
                known.add((getattr(n, "raw", "") or "").strip())
        except Exception:
            known = set()
        if body in known:
            self._save(last)
            return 0
        try:
            node = book.add(EVENT, 1.0, "", raw=body, who="")
        except Exception:
            return 0                       # 落不上，游标留着，下一轮再试
        # 显式连到这一批：反思是事后落的，靠时间窗连不上自己那几条依据
        try:
            for src in batch:
                book._link(node, src, 1.0)
            book._append([node] + batch)
        except Exception:
            pass
        self._save(last)
        if callable(link):
            try:
                link(node)
            except Exception:
                pass                       # 向量连不上不影响节点已经落了
        return 1
