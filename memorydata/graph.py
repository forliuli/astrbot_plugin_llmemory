# -*- coding: utf-8 -*-
"""记忆库的图画层：近邻、边、检索、加固、衰减。

最初由一份脚本层原样搬来，只把「行号 + chunks.jsonl」换成「节点 id + NodeBook」。
顺序和公式一条没改：
  建边   语义近邻（cosine >= min_sim，取 topk）+ 时间相邻（gap <= tmax）
  截度   每颗节点最多 max_deg 条，留最粗的
  检索   sift = sim * scale 筛池 -> 乘时间衰减 -> 关键词加成 -> 种子过阈值封顶
         -> 一跳扩散 contrib = score[src] * w * alpha -> 双通路出 k+flip 条
  加固   本次命中的头部两两加边，inc
  衰减   边 w *= exp(-age / tau)，枢纽（度 >= hub_deg）不动，过 floor 只降不到零
"""

import json
import math
import os
import time

import numpy as np


def _f(env, default):
    try:
        return float(os.environ.get(env, default))
    except Exception:
        return default


def _tok(text):
    """很粗的分词：中文按字，别的按词。只用来给关键词加一点分。"""
    out = []
    buf = ""
    for ch in str(text):
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                out.append(buf.lower())
                buf = ""
            out.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf:
                out.append(buf.lower())
                buf = ""
    if buf:
        out.append(buf.lower())
    return [t for t in out if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]


class Graph:
    """节点表按 id 对齐；V 与 ids 逐行一致；边用 (a,b) 位置的扁平数组存。"""

    def __init__(self, path="", dim=512):
        self.path = path
        self.dim = int(dim)
        self.ids = []            # 行号 -> 节点 id
        self.row = {}            # 节点 id -> 行号
        self.ts = []             # 行 -> 时间戳
        self.scale = []          # 行 -> 节点 scale（当权重用）
        self._vecs = []          # 行 -> 一条向量（None 表示还没算）
        self._V = None           # 合并缓存
        self._dirty = True
        self.edges = {}          # (a, b) a<b -> w

    # ---------- 存取 ----------

    def load(self):
        if not self.path or not os.path.isfile(self.path):
            return False
        try:
            z = np.load(self.path, allow_pickle=False)
            self.ids = [str(x) for x in z["ids"]]
            self.row = {k: i for i, k in enumerate(self.ids)}
            self.ts = list(z["ts"].astype(np.float64))
            self.scale = list(z["scale"].astype(np.float32))
            V = z["V"].astype(np.float32)
            self._vecs = [V[i] for i in range(V.shape[0])]
            self.dim = V.shape[1] if V.ndim == 2 else self.dim
            if "esrc" in z:
                es = z["esrc"].astype(np.int32)
                ed = z["edst"].astype(np.int32)
                ew = z["ew"].astype(np.float32)
                self.edges = {(min(int(a), int(b)), max(int(a), int(b))): float(w)
                              for a, b, w in zip(es, ed, ew)}
            self._dirty = True
            return True
        except Exception:
            return False

    def save(self):
        if not self.path:
            return False
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            V = self.matrix()
            ek = list(self.edges.keys())
            es = np.fromiter((a for a, b in ek), dtype=np.int32, count=len(ek))
            ed = np.fromiter((b for a, b in ek), dtype=np.int32, count=len(ek))
            ew = np.fromiter((self.edges[k] for k in ek), dtype=np.float32, count=len(ek))
            tmp = self.path + ".tmp"
            np.savez_compressed(tmp, ids=np.array(self.ids, dtype="<U64"),
                                ts=np.array(self.ts, dtype=np.float64),
                                scale=np.array(self.scale, dtype=np.float32),
                                V=V if V is not None else np.zeros((0, self.dim), np.float32),
                                esrc=es, edst=ed, ew=ew)
            os.replace(tmp + ".npz", self.path)
            return True
        except Exception:
            return False

    # ---------- 行 ----------

    def matrix(self):
        if self._V is None or self._dirty:
            n = len(self._vecs)
            if n == 0:
                self._V = np.zeros((0, self.dim), dtype=np.float32)
            else:
                dim = self._vecs[0].shape[-1] if self._vecs[0] is not None else self.dim
                V = np.zeros((n, dim), dtype=np.float32)
                for i, v in enumerate(self._vecs):
                    if v is not None:
                        V[i] = v
                self._V = V
                self.dim = dim
            self._dirty = False
        return self._V

    def add(self, nid, vec, ts=None, scale=1.0, k=8, min_sim=0.50,
            tmax=3 * 3600, w_time=0.6, max_deg=16):
        """追加一行，顺手给新行连边（语义近邻 + 与上一行的时间相邻）。"""
        nid = str(nid)
        if nid in self.row:
            return self.row[nid]
        r = len(self.ids)
        self.ids.append(nid)
        self.row[nid] = r
        self.ts.append(float(ts or time.time()))
        self.scale.append(float(scale or 1.0))
        self._vecs.append(None if vec is None else np.asarray(vec, dtype=np.float32))
        self._dirty = True

        if vec is None:
            return r
        v = np.asarray(vec, dtype=np.float32)
        V = self.matrix()
        if r > 0:
            sims = V[:r] @ v
            m = min(k, r)
            idx = np.argpartition(-sims, m - 1)[:m]
            for j in idx:
                s = float(sims[j])
                if s >= min_sim:
                    self._put(r, int(j), s)
        # 时间相邻：找 ts 最近的前一行
        if r > 0:
            prev = r - 1
            gap = self.ts[r] - self.ts[prev]
            if 0 <= gap <= tmax:
                self._put(r, prev, w_time)
        if max_deg:
            self._cap(r, max_deg)
        return r

    def _put(self, a, b, w):
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if w > self.edges.get(key, 0.0):
            self.edges[key] = float(w)

    def _cap(self, r, max_deg):
        mine = [(k, w) for k, w in self.edges.items() if r in k]
        if len(mine) <= max_deg:
            return
        mine.sort(key=lambda x: -x[1])
        for k, _ in mine[max_deg:]:
            self.edges.pop(k, None)

    # ---------- 全量建边（照 hebb.build） ----------

    def rebuild(self, nodes, k=8, min_sim=0.50, tmax=3 * 3600,
                w_time=0.6, w_path=0.8, max_deg=16):
        """nodes: [(id, ts, scale, kind)]，按当前全表重算边。

        行号以图自己为准：外面那张表可能缺行、顺序也对不上，先按 id 对回来，
        对不上的行不参与建边，也不许占别人的位置。
        """
        self.edges = {}
        if not nodes:
            return 0
        info = {}
        for nid, t, s, kind in nodes:
            info[str(nid)] = (float(t or 0), float(s or 1.0), str(kind or ""))
        rows = [i for i, nid in enumerate(self.ids) if nid in info]
        n = len(rows)
        if n == 0:
            return 0
        M = self.matrix()[rows]                     # 只取参与建边的那几行
        ts = np.array([info[self.ids[r]][0] for r in rows], dtype=np.float64)
        has = [self._vecs[r] is not None for r in rows]
        # 1) 语义近邻
        m = min(k, n - 1)
        if m > 0:
            for p in range(n):
                if not has[p]:
                    continue
                sims = M @ M[p]
                sims[p] = -1.0
                idx = np.argpartition(-sims, m - 1)[:m]
                for j in idx:
                    s = float(sims[j])
                    if s >= min_sim:
                        self._put(rows[p], rows[int(j)], s)
        # 2) 时间相邻
        order = np.argsort(ts)
        for a, b in zip(order[:-1], order[1:]):
            gap = ts[b] - ts[a]
            if 0 <= gap <= tmax:
                self._put(rows[int(a)], rows[int(b)], w_time)
        # 3) 同源/同类相邻
        byp = {}
        for p, r in enumerate(rows):
            byp.setdefault(info[self.ids[r]][2], []).append(p)
        if w_path:
            for _, lst in byp.items():
                for x, y in zip(lst[:-1], lst[1:]):
                    self._put(rows[x], rows[y], w_path)
        # 4) 截度
        if max_deg:
            adj = {}
            for (a, b), w in sorted(self.edges.items(), key=lambda kv: -kv[1]):
                if len(adj.get(a, ())) < max_deg:
                    adj.setdefault(a, []).append((b, w))
                if len(adj.get(b, ())) < max_deg:
                    adj.setdefault(b, []).append((a, w))
            keep = set()
            for a, lst in adj.items():
                for b, _ in lst:
                    keep.add((a, b) if a < b else (b, a))
            self.edges = {k: self.edges[k] for k in keep}
        return len(self.edges)

    def _arrays(self):
        ek = list(self.edges.keys())
        src = np.fromiter((a for a, b in ek), dtype=np.int32, count=len(ek))
        dst = np.fromiter((b for a, b in ek), dtype=np.int32, count=len(ek))
        w = np.fromiter((self.edges[k] for k in ek), dtype=np.float32, count=len(ek))
        return src, dst, w

    # ---------- 检索（照 hebb.retrieve2） ----------

    def retrieve(self, qvec, texts=None, k=8, pool=400, floor=None, alpha=None,
                 theta=None, max_seed=None, flip=None, use_kw=True, use_time=True,
                 lam=None, tau=None, kw_boost=None, now=None, node_texts=None):
        floor = _f("HEBB_FLOOR", 0.30) if floor is None else floor
        alpha = _f("HEBB_ALPHA", 0.10) if alpha is None else alpha
        theta = _f("HEBB_THETA", 0.60) if theta is None else theta
        max_seed = int(_f("HEBB_SEED", 20)) if max_seed is None else max_seed
        flip = int(_f("HEBB_FLIP", 5)) if flip is None else flip
        kw_boost = _f("HEBB_KW", 0.12) if kw_boost is None else kw_boost
        lam = _f("HEBB_LAM", 0.35) if lam is None else lam
        tau = _f("HEBB_TAU", 30 * 86400.0) if tau is None else tau
        now = time.time() if now is None else now

        V = self.matrix()
        n = V.shape[0]
        if n == 0 or qvec is None:
            return []

        sim = V @ np.asarray(qvec, dtype=np.float32)
        wt = np.ones(n, dtype=np.float32)   # 照 hebb 的 m["w"]：没强弱之分就一律 1.0
        sift = sim * wt

        cand = np.argsort(-sift)[:pool]
        cand = cand[sift[cand] >= floor]
        if len(cand) == 0:
            return []

        if use_time:
            ts = np.array(self.ts, dtype=np.float64)
            has = ts > 0
            age = np.where(has, np.maximum(0.0, now - ts), 0.0) / tau
            dec = np.exp(-lam * np.minimum(age, 60.0))
        else:
            dec = np.ones(n, dtype=np.float32)

        score = sift * dec
        kw = np.zeros(n, dtype=np.float32)
        if use_kw and texts:
            qt = _tok(texts)
            if qt:
                for i in cand:
                    t = node_texts[i] if node_texts is not None and i < len(node_texts) else ""
                    kw[i] = sum(1 for x in qt if x in t) / len(qt)
                score += kw_boost * kw * (sift > floor)

        top = int(max(cand, key=lambda i: score[i]))
        cand = np.array(sorted(cand, key=lambda i: -score[i]))
        norm = score[cand] / max(score[top], 1e-6)
        seeds = cand[norm >= theta][:max_seed]
        sel = np.zeros(n, dtype=bool)
        sel[seeds] = True

        src, dst, w = self._arrays()
        boost = np.zeros(n, dtype=np.float32)
        if seeds.size and len(src):
            m = sel[src]
            if m.any():
                dd = dst[m]
                cc = (score[src[m]] * w[m] * alpha).astype(np.float32)
                o = np.argsort(dd, kind="stable")
                dd, cc = dd[o], cc[o]
                uniq, start = np.unique(dd, return_index=True)
                boost[uniq] = np.maximum.reduceat(cc, start)

        total = score + boost
        total[sel] = score[sel]

        base_out = [int(i) for i in cand[:k]]
        spread = np.argsort(-total)[: k * 4]
        flipped = [int(i) for i in spread if int(i) not in base_out and boost[int(i)] > 0][:flip]

        out = []
        for i in base_out:
            out.append({"node": i, "id": self.ids[i], "score": float(total[i]),
                        "base": float(sift[i]), "dec": float(dec[i]),
                        "boost": float(boost[i]), "kw": float(kw[i]),
                        "sim": float(sim[i]), "source": "base"})
        for i in flipped:
            out.append({"node": i, "id": self.ids[i], "score": float(total[i]),
                        "base": float(sift[i]), "dec": float(dec[i]),
                        "boost": float(boost[i]), "kw": float(kw[i]),
                        "sim": float(sim[i]), "source": "hebbian"})
        return out

    # ---------- 加固（照 hebb.reinforce）与衰减 ----------

    def reinforce(self, out, inc=0.05):
        nodes = [o["node"] for o in out if "node" in o]
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                self._put(nodes[a], nodes[b], inc)
        return self

    def decay(self, tau=None, floor=0.05, hub_deg=32, now=None, since=None):
        """边 w *= exp(-age/tau)。age 取两端节点里较新的时间；枢纽的边不动；只降不删。

        给了 since 就按「离那一趟多久」算，不再按节点自己的岁数——隔几个钟头跑一次
        才不会一天压掉好几天的量。
        """
        tau = _f("HEBB_EDGE_TAU", tau if tau is not None else 30 * 86400.0)
        now = time.time() if now is None else now
        if not self.edges:
            return 0
        deg = {}
        for a, b in self.edges:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        ts = self.ts
        out = {}
        changed = 0
        for (a, b), w in self.edges.items():
            if deg.get(a, 0) >= hub_deg or deg.get(b, 0) >= hub_deg:
                out[(a, b)] = w
                continue
            if since is not None:
                age = max(0.0, now - since)
            else:
                age = max(0.0, now - max(ts[a] if a < len(ts) else 0,
                                         ts[b] if b < len(ts) else 0))
            f = math.exp(-age / max(tau, 1.0))
            nw = max(floor, w * f) if w > floor else w
            if nw != w:
                changed += 1
            out[(a, b)] = nw
        self.edges = out
        return changed

    # ---------- 去重（照 hebb.dedup，文本恰好相同才并） ----------

    @staticmethod
    def dedup(texts):
        """同一文本只留一个代表节点：组内 ts 最早的那颗。返回 {行号: 代表行号}。"""
        groups = {}
        for i, t in enumerate(texts):
            groups.setdefault(t, []).append(i)
        rep_of = {}
        for _, idxs in groups.items():
            r = idxs[0]
            for i in idxs:
                rep_of[i] = r
        return rep_of
