"""自唤醒。

没人说话的时候它自己待着：隔一阵走一趟（神游），再看要不要开口。
开口有门槛，门槛不够就不说——自唤醒不是定时报时。

1. 碎片得是这段时间新落的；
2. 得跟当下心情对得上，对不上宁可沉默；
3. 静默时段不说，刚说过不说，一小时、一天都有数；
4. 六维太沉的时候更少开口。

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

TICK = 1800.0          # 多久醒一次
QUIET = (1, 7)         # 静默时段：凌晨一点到七点
PER_HOUR = 2
PER_DAY = 6
MIN_GAP = 1800.0       # 两次开口至少隔半小时
FRESH_MINUTES = 240.0  # 四小时内落的碎片才算新
KEEP_SENT = 24

STATE_NAME = os.path.join("plugin_data", "状态", "wake.json")
TARGET_NAME = os.path.join("plugin_data", "状态", "wake_target.txt")


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
        self.per_day = int(_num(raw, "wake_per_day", PER_DAY, 0, 96))
        self.min_gap = _num(raw, "wake_min_gap", MIN_GAP, 60, 86400)
        self.fresh_minutes = _num(raw, "wake_fresh_minutes", FRESH_MINUTES, 10, 1440)
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
        bias = self._bias()
        if bias <= 0:
            return False
        gap = self.min_gap * bias
        if self.sent and now - self.sent[-1] < gap:
            return False
        hour_ago = [t for t in self.sent if now - t < 3600]
        if len(hour_ago) >= max(1, int(round(self.per_hour / bias))):
            return False
        day_ago = [t for t in self.sent if now - t < 86400]
        if len(day_ago) >= max(1, int(round(self.per_day / bias))):
            return False
        return True

    def pick(self, now: float) -> dict | None:
        """这段里新落的、又跟当下心情对得上的那一句。挑不出就不说。"""
        frags = M.fresh(self.muse_path, since_ts=now - self.fresh_minutes * 60, limit=12)
        if not frags:
            return None
        mood = self._mood()
        hit = [f for f in frags if M.matches(f, mood)]
        if not hit:
            return None
        fresh = [f for f in hit if str(f.get("start") or "") not in self.said_starts]
        return (fresh or hit)[-1]

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
            logger.warning(f"[llmemory] 神游没走成：{e}")

    def target(self) -> str:
        umo = getattr(self.plugin, "last_umo", "") or ""
        if umo:
            return umo
        try:
            with open(TARGET_NAME, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    async def say(self, text: str) -> bool:
        """往最近的会话发一句。发不出去就当这次没醒，不改账。"""
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
                logger.warning(f"[llmemory] 发消息的零件导不进来：{e}")
                return False
        try:
            await context.send_message(umo, MessageChain([Plain(text)]))
            return True
        except Exception as e:
            logger.warning(f"[llmemory] 自唤醒发不出去：{e}")
            return False

    async def tick(self) -> bool:
        now = time.time()
        if self.drift_enable and now - self.last_drift >= TICK * 0.9:
            await asyncio.to_thread(self.drift, now)
        if not self.wake_enable:
            return False
        if self.is_quiet(now) or not self._allowed(now):
            return False
        item = self.pick(now)
        if not item:
            return False
        if not await self.say(item.get("line", "")):
            return False
        self.sent.append(now)
        self.said_starts = ([str(item.get("start") or "")] + self.said_starts)[:8]
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
                logger.warning(f"[llmemory] 自唤醒这一轮出错：{e}")
            await asyncio.sleep(TICK * random.uniform(0.9, 1.1))

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
