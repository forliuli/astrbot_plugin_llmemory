"""自唤醒。

没人说话的时候它自己待着：隔一阵走一趟（神游），再看要不要开口。
开口有门槛，门槛不够就不说——自唤醒不是定时报时。

1. 碎片得是这段时间新落的；
2. 神游走到的那颗得是强激活的——分量、边、新旧算出来的分要排进全书前列，
   跟当下心情对得上再抬一点；走到的是颗不响的节点，这一趟就算了；
3. 静默时段不说，刚说过不说，一小时、一天都有数；
4. 六维太沉的时候更少开口；
5. 说出去的那句由模型写：强节点和最近几轮对话一起喂进去，
   神游拼的轨迹句只当材料，不原样发。

只往最近说过话的那个会话发。没会话就等着，不猜。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time

from . import muse as M
from .nodes import load_book

try:
    from astrbot.api import logger
except Exception:                                   # 单独跑脚本时
    import logging
    logger = logging.getLogger("llmemory.wake")

TICK = 60.0            # 多久醒一次（默认一分钟，面板可改）
QUIET = (1, 7)         # 静默时段：凌晨一点到七点
PER_HOUR = 3           # 一小时最多开口几次，就这一道闸
FRESH_MINUTES = 240.0  # 四小时内落的碎片才算新
KEEP_SENT = 24
STRONG_SCALE = 1.6     # 走到的这颗，激活强度过了它才算强（|scale|，跟神游分高低同一把尺）
RECENT_NODES = 4       # 拼上下文时带最近几颗的原话
MAX_SAY = 90           # 模型写多了就切

STATE_NAME = os.path.join("plugin_data", "状态", "wake.json")
TARGET_NAME = os.path.join("plugin_data", "状态", "wake_target.txt")

SYSTEM = (
    "你正在和对方私聊。这会儿没人开口，你自己待着，"
    "脑子里浮起一件旧事，于是先说一句。"
    "只写你要发出去的那一句话：二十来个字，像平时说话那种口气；"
    "不复述旧事，不解释为什么想起，不总结，不问「你怎么了」；"
    "不加引号，不写括号里的动作，不用称呼。"
)


def _num(raw: dict, key: str, default: float, lo: float, hi: float) -> float:
    """面板上的数不一定老实，夹一下再用。"""
    try:
        value = float(raw.get(key, default))
    except (TypeError, ValueError):
        return default
    if value != value:                              # NaN
        return default
    return max(lo, min(hi, value))


def _dim(state, key: str, default: float = 50.0) -> float:
    """六维读数。字段名对不上就当平，不为这个报错。"""
    value = getattr(state, key, None)
    if value is None and isinstance(state, dict):
        value = state.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_cn(code: str) -> str:
    """事件代号翻成人话。表换了或认不出来就原样给。"""
    table = getattr(M, "_EVENT_CN", {}) or {}
    return str(table.get(str(code or ""), code or ""))


def _mood_core(line: str) -> str:
    """「心情：现在有劲」剥成「有劲」。muse 那边有就用那边的，别两套写法。"""
    fn = getattr(M, "_mood_core", None)
    if fn is not None:
        try:
            return fn(line)
        except Exception:
            pass
    return str(line or "").strip(" 　。.，,")


class Waker:
    """一个循环：神游 + 开口。plugin 要带 state / cfg / raw_config / context。"""

    def __init__(self, plugin, nodes_path: str = "", muse_path: str = "",
                 state_path: str = ""):
        self.plugin = plugin
        self.nodes_path = nodes_path or os.path.join("plugin_data", "nodes.json")
        self.muse_path = muse_path or M.DEFAULT_PATH
        self.state_path = state_path or STATE_NAME
        self.sent = []                              # 开口的时刻，跨重启留着
        self.said_starts = []
        self.last_drift = 0.0
        self._task = None
        self._stopped = False
        raw = getattr(plugin, "raw_config", {}) or {}
        self.quiet = (int(_num(raw, "wake_quiet_start", QUIET[0], 0, 23)),
                      int(_num(raw, "wake_quiet_end", QUIET[1], 0, 23)))
        self.per_hour = int(_num(raw, "wake_per_hour", PER_HOUR, 0, 24))
        self.fresh_minutes = _num(raw, "wake_fresh_minutes", FRESH_MINUTES, 10, 1440)
        self.strong_scale = _num(raw, "wake_strong_scale", STRONG_SCALE, 0.1, 5.0)
        self.tick_seconds = _num(raw, "wake_drift_minutes", TICK / 60.0, 1.0, 1440.0) * 60.0
        self.drift_enable = bool(raw.get("drift_enable", True))
        self.wake_enable = bool(raw.get("wake_enable", False))   # 走不走 ≠ 说不说
        self._load()

    # ---- 存 ------------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.sent = [float(t) for t in data.get("sent", [])][-KEEP_SENT:]
        except (OSError, ValueError, TypeError, AttributeError):
            self.sent = []

    def _save(self) -> None:
        try:
            folder = os.path.dirname(self.state_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"sent": self.sent[-KEEP_SENT:]}, f)
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    # ---- 判断 ----------------------------------------------------------

    def _hour(self, now: float) -> int:
        return time.localtime(now).tm_hour

    def is_quiet(self, now: float) -> bool:
        start, end = self.quiet
        hour = self._hour(now)
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end            # 跨零点

    def _mood(self) -> str:
        try:
            from .affect import describe
            return describe(getattr(self.plugin, "state", None)) or ""
        except Exception:
            return ""

    def _bias(self) -> float:
        """心情沉的时候把门槛抬高，黏的时候松一点。小于 1 表示这次别开口。"""
        state = getattr(self.plugin, "state", None)
        drive, stable = _dim(state, "D"), _dim(state, "S")
        close, comfort = _dim(state, "O", 20.0), _dim(state, "E", 40.0)
        if comfort < 25 and drive < 30:
            return 0.0                               # 太沉，先不说话
        scale = 1.0
        if comfort < 35:
            scale *= 1.5
        if drive < 30:
            scale *= 1.4
        if stable < 30:
            scale *= 1.3
        if close >= 45:
            scale *= 0.6
        return scale

    def _allowed(self, now: float) -> bool:
        """就一道闸：滚动的这一小时里说满几条，剩下的时间闭嘴。"""
        bias = self._bias()
        if bias <= 0:
            return False
        cap = max(1, int(round(self.per_hour / bias)))
        hour_ago = [t for t in self.sent if now - t < 3600]
        return len(hour_ago) < cap

    def pick(self, now: float) -> dict | None:
        """这段时间新落的碎片里，神游走到的那颗够强就说；不够强这次就不说。"""
        frags = M.fresh(self.muse_path, since_ts=now - self.fresh_minutes * 60, limit=12)
        if not frags:
            return None
        try:
            book = load_book(self.nodes_path)
        except Exception as e:
            logger.warning(f"[llmemory] 自唤醒读取节点库失败：{e}")
            return None
        if not len(book):
            return None
        mood = self._mood()
        for item in reversed(frags):                 # 从最新的往回找
            start = str(item.get("start") or "")
            if start and start in self.said_starts:
                continue
            node = book.nodes.get(str(item.get("end") or ""))
            if node is None:                         # 那颗被裁掉了，换一颗
                continue
            if abs(getattr(node, "scale", 0.0)) < self.strong_scale:
                continue
            return {"frag": item, "node": node, "book": book}
        return None

    # ---- 动 ------------------------------------------------------------

    def drift(self, now: float) -> None:
        """走一趟，把碎片落盘。走不成（没节点）就算了，不报错。"""
        if not self.drift_enable:
            return
        try:
            book = load_book(self.nodes_path)
            if not len(book):
                return
            item = M.run_once(book, self.muse_path, mood=self._mood(), now_ts=now)
            if item:
                self.last_drift = now
        except Exception as e:
            logger.warning(f"[llmemory] 神游执行失败：{e}")

    def target(self) -> str:
        umo = getattr(self.plugin, "last_umo", "") or ""
        if umo:
            return umo
        try:
            with open(TARGET_NAME, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    def _recent(self, book) -> str:
        """最近几轮的原话，给他当上下文。没有就空着。"""
        items = sorted(book.nodes.values(), key=lambda n: n.ts, reverse=True)[:RECENT_NODES]
        lines = []
        for node in reversed(items):
            raw = str(node.raw or "").strip()
            if not raw:
                continue
            lines.append(raw.replace("\n", " / ")[:160])
        return "\n".join(lines)

    def brief(self, frag: dict, node, book) -> str:
        """把这一趟的材料摊开：此刻、走到的那颗、最近说过的话。"""
        stamp = time.strftime("%m-%d %H:%M", time.localtime(time.time()))
        parts = [f"【此刻】{stamp}，心情：{self._mood() or '说不上来'}。"]
        parts.append("【浮上来的一件旧事】")
        event = _event_cn(node.event)
        if event:
            parts.append(f"那时候：{event}")
        mood = _mood_core(node.mood)
        if mood:
            parts.append(f"那时候的心情：{mood}")
        raw = str(node.raw or "").strip()
        if raw:
            parts.append("当时的原话：" + raw[:200].replace("\n", " / "))
        hops = [e for e in (frag.get("events") or []) if e]
        if len(hops) > 1:
            parts.append("路上还走过：" + " → ".join(_event_cn(e) for e in hops))
        recent = self._recent(book)
        if recent:
            parts.append("【最近几轮说过的话】\n" + recent)
        parts.append("写一句发过去的话。")
        return "\n".join(parts)

    @staticmethod
    def _clean(text: str) -> str:
        """模型写回来的东西收一收：取第一行，剥引号，太长就切。"""
        text = str(text or "").strip()
        if not text:
            return ""
        for chunk in text.splitlines():
            chunk = chunk.strip().strip("\"'“”「」'").strip()
            if chunk:
                text = chunk
                break
        return text[:MAX_SAY]

    async def compose(self, frag: dict, node, book) -> str:
        """过模型再开口。没有模型入口就不说，绝不退回轨迹原句。"""
        gen = getattr(self.plugin, "_gen_text", None)
        if gen is None:
            logger.warning("[llmemory] 自唤醒未配置模型，本轮跳过")
            return ""
        try:
            text = await gen(SYSTEM, self.brief(frag, node, book))
        except Exception as e:
            logger.warning(f"[llmemory] 自唤醒模型调用失败：{e}")
            return ""
        return self._clean(text)

    async def say(self, text: str) -> bool:
        """往最近的会话发一句。发不出去就当这次没醒，不改账。"""
        text = str(text or "").strip()
        if not text:
            return False
        umo = self.target()
        if not umo:
            return False
        context = getattr(self.plugin, "context", None)
        if context is None:
            return False
        try:
            from astrbot.api.event import MessageChain
            from astrbot.api.message_components import Plain
        except Exception:
            try:
                from astrbot.core.message.message_event_result import MessageChain
                from astrbot.core.message.components import Plain
            except Exception as e:
                logger.warning(f"[llmemory] 消息发送模块导入失败：{e}")
                return False
        try:
            await context.send_message(umo, MessageChain([Plain(text)]))
            return True
        except Exception as e:
            logger.warning(f"[llmemory] 自唤醒消息发送失败：{e}")
            return False

    async def tick(self) -> bool:
        now = time.time()
        if self.drift_enable and now - self.last_drift >= self.tick_seconds * 0.9:
            await asyncio.to_thread(self.drift, now)
        if not self.wake_enable:
            return False
        if self.is_quiet(now) or not self._allowed(now):
            return False
        got = self.pick(now)
        if not got:
            return False
        text = await self.compose(got["frag"], got["node"], got["book"])
        if not text:
            return False
        if not await self.say(text):
            return False
        self.sent.append(now)
        self.said_starts = ([str(got["frag"].get("start") or "")] + self.said_starts)[:8]
        self._save()
        return True

    # ---- 起停 ----------------------------------------------------------

    async def loop(self) -> None:
        while not self._stopped:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:                  # 循环不能死
                logger.warning(f"[llmemory] 自唤醒本轮出错：{e}")
            await asyncio.sleep(self.tick_seconds * random.uniform(0.9, 1.1))

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopped = False
        self._task = asyncio.create_task(self.loop())

    async def stop(self) -> None:
        self._stopped = True
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._save()


def write_target(umo: str) -> None:
    """记下最近说过话的会话，重启后还认得路。"""
    if not umo:
        return
    try:
        folder = os.path.dirname(TARGET_NAME)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(TARGET_NAME, "w", encoding="utf-8") as f:
            f.write(umo)
    except OSError:
        pass
