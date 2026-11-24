# -*- coding: utf-8 -*-
"""记忆库取回：用户这句话 -> 该拿哪几颗节点出来。

顺序照本地那套走：
  一句话过编码器 -> 图里检索（近邻 + 时间衰减 + 关键词 + 一跳扩散）
  -> 命中头部两两加边（reinforce）-> 按 id 对回 NodeBook（被裁掉的丢掉）
没有模型、检索不出来就退回 book.hot()，不硬撑着给空。
"""

import os
import time

import numpy as np

from .nodes import Node


# 事件类型 -> 中文短语。落向量和检索都读这一个口子，
# 否则 add() 编码 event、query() 比 event+raw，落进去也搜不出来。
_EVENT_CN = {
    "praise": "被夸那下",
    "criticism": "被挑那下",
    "bonding": "凑近那下",
    "stress": "压过来那下",
    "relaxation": "松下来那下",
    "talk": "说了会儿话",
}


def node_text(node):
    """一颗节点该被读成什么话：中文事件短语 + 原话。"""
    if node is None:
        return ""
    ev = (node.event or "").strip()
    raw = (node.raw or "").strip()
    head = _EVENT_CN.get(ev, ev)
    return " ".join([p for p in (head, raw) if p]).strip()


def _ago(ts, now=None):
    now = time.time() if now is None else now
    d = max(0.0, now - float(ts or 0))
    if d < 3600:
        return "%d分钟前" % max(1, int(d // 60))
    if d < 86400:
        return "%d小时前" % int(d // 3600)
    return "%d天前" % int(d // 86400)


class Recall:
    def __init__(self, book, graph=None, enc=None):
        self.book = book
        self.graph = graph
        self.enc = enc

    # ---- 写：落一颗时顺带算向量、连边 ----

    def add(self, node):
        """节点刚落进 NodeBook 之后调。算不出来就只落节点，不拦。"""
        if node is None or self.graph is None:
            return False
        try:
            vec = self.enc.one(node_text(node)) if self.enc is not None else None
            self.graph.add(node.id, vec, ts=node.ts, scale=node.scale)
            return vec is not None
        except Exception:
            return False

    def backfill(self, batch=16, limit=0):
        """历史节点补向量。返回补上的颗数。"""
        if self.graph is None or self.enc is None or not self.enc.ready:
            return 0
        missing = [n for n in self.book.nodes.values() if n.id not in self.graph.row]
        if limit:
            missing = missing[:limit]
        if not missing:
            return 0
        done = 0
        for i in range(0, len(missing), batch):
            part = missing[i:i + batch]
            texts = [node_text(n) for n in part]
            try:
                V = self.enc.encode(texts, batch=batch, quiet=True)
            except Exception:
                break
            if V is None:
                break
            for n, v in zip(part, V):
                try:
                    self.graph.add(n.id, v, ts=n.ts, scale=n.scale)
                    done += 1
                except Exception:
                    continue
        self.graph.save()
        return done

    def rebuild(self):
        """全表重算边。慢活儿，后台跑。"""
        if self.graph is None:
            return 0
        try:
            rows = [(n.id, n.ts, n.scale, "") for n in self.book.nodes.values()]
            got = self.graph.rebuild(rows)
            self.graph.save()
            return got
        except Exception:
            return 0

    # ---- 读：一句话取回 ----

    def query(self, text, k=6, now=None):
        """返回 [(node, score, source)]，只含还活着的节点。"""
        text = (text or "").strip()
        if not text:
            return []
        if self.graph is None or self.enc is None or not self.enc.ready:
            return [(n, 1.0, "hot") for n in self.book.hot(limit=k, pool=200)]

        try:
            qvec = self.enc.one(text)
            node_texts = []
            for nid in self.graph.ids:
                nd = self.book.nodes.get(nid)
                node_texts.append(node_text(nd))
            out = (self.graph.retrieve(qvec, texts=text, node_texts=node_texts, k=k, now=now)
                   if qvec is not None else [])
        except Exception:
            out = []

        got, seen = [], set()
        for o in out:
            nid = o.get("id")
            node = self.book.nodes.get(nid)
            if node is None or nid in seen:
                continue        # 已裁剪/重复，对不上就丢
            seen.add(nid)
            got.append((node, float(o.get("score", 0.0)), o.get("source", "base")))
        if got:
            try:
                self.graph.reinforce([{"node": self.graph.row[n.id]} for n, _, _ in got if n.id in self.graph.row])
            except Exception:
                pass
        else:
            got = [(n, 1.0, "hot") for n in self.book.hot(limit=k, pool=200)]
        return got

    def render(self, items, now=None, max_nodes=6, max_chars=700):
        """拼成给提示的一段。空就返回空串，不塞占位。"""
        if not items:
            return ""
        now = time.time() if now is None else now
        lines, used = [], 0
        for node, _, _ in items[:max_nodes]:
            body = (node.event or "").strip()
            if node.mood:
                body += "（" + node.mood.strip()[:40] + "）"
            if node.raw:
                body += "｜" + node.raw.strip().replace("\n", " ")[:60]
            line = "- " + _ago(node.ts, now) + "：" + body
            used += len(line)
            if used > max_chars:
                break
            lines.append(line)
        if not lines:
            return ""
        return "【想起】\n" + "\n".join(lines)
