"""llmemory 插件入口。

T0 只做一件事的闭环：收一句话 → 判 → 落到六维 → 存盘。
注入那层只给一句，不抄日志，也不替模型组织措辞。
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.message import TextPart

try:                                      # 按包加载（AstrBot 走这条）
    from .llmemory.affect import (
        apply_judgement, chat, describe, judge_model, judge_tuple, load_config,
    )
    from .llmemory import paths as P
    from .llmemory.graph import Graph
    from .llmemory.nodes import load_book
    from .llmemory.recall import Recall
    from .llmemory.reflect import Reflector
    from .llmemory.vec import Encoder, deps_error
    from .llmemory.model_fetch import fetch_in_background
    from .llmemory.wake import Waker, write_target
    from .llmemory.writer import Writer, _read
except ImportError:                       # 插件目录在 sys.path 上时
    from llmemory.affect import (
        apply_judgement, chat, describe, judge_model, judge_tuple, load_config,
    )
    from llmemory import paths as P
    from llmemory.graph import Graph
    from llmemory.nodes import load_book
    from llmemory.recall import Recall
    from llmemory.reflect import Reflector
    from llmemory.vec import Encoder, deps_error
    from llmemory.model_fetch import fetch_in_background
    from llmemory.wake import Waker, write_target
    from llmemory.writer import Writer, _read

PLUGIN_NAME = "astrbot_plugin_llmemory"
_PLUGIN_ROOT = os.path.dirname(os.path.abspath(__file__))

# 这一句落成了哪一下。给模型的是人话，不是事件名。
_EVENT_CN = {
    "praise": "对方这句是夸过来的",
    "criticism": "对方这句是挑着的",
    "bonding": "对方这句是往近处凑的",
    "stress": "对方这句是压过来的",
    "relaxation": "对方这句是松下来的",
}


def _vec_urls(raw_conf) -> list:
    """向量模型地址：逗号或换行分隔，按顺序试，去重。"""
    val = raw_conf.get("vec_model_urls") if hasattr(raw_conf, "get") else ""
    if isinstance(val, (list, tuple)):
        items = [str(x) for x in val]
    else:
        items = str(val or "").replace("\r", "\n").replace(",", "\n").split("\n")
    out = []
    for it in items:
        it = it.strip()
        if it and it not in out:
            out.append(it)
    return out


def _vec_sha(raw_conf) -> str:
    return str(raw_conf.get("vec_model_sha256") or "").strip()


def node_mood(state) -> str:
    """该存的那一句。只存反读，六个数不进节点。"""
    line = describe(state)
    return f"心情：{line}" if line else ""


def _data_dir() -> str:
    """插件的数据目录：data/plugin_data/<插件名>，由框架给。
    面板填了 data_dir 或环境变量 LLMEMORY_DATA_DIR 就听它的。"""
    return P.data_dir()


MOOD_OUT = os.path.join(_data_dir(), "状态", "llmemory_mood.json")


def write_mood(state, judgement=None, out_path: str | None = None) -> str:
    """这一刻的心情，带时间戳落盘。要用的时候照这个抄一句。"""
    line = node_mood(state)
    if not line:
        return ""
    event_type, scale = "", 0.0
    if judgement:
        try:
            event_type, scale = str(judgement[0]), float(judgement[1])
        except Exception:
            pass
    payload = {"ts": time.time(), "line": line,
               "event": event_type, "scale": scale}
    out = out_path or MOOD_OUT
    try:
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, out)
    except Exception as e:
        logger.warning(f"[llmemory] 心情没落上：{e}")
        return ""
    return line


@register(
    PLUGIN_NAME,
    "",
    "情绪与内分泌层：把一句话判成五个事件之一，落到六维状态上，跨轮存盘。不接模型也能跑。",
    "0.1.0",
    "",
)
def _speaker(event) -> str:
    """谁说的。取不到就空，别为一个名字崩。"""
    fn = getattr(event, "get_sender_name", None)
    if callable(fn):
        try:
            name = str(fn() or "").strip()
        except Exception:
            name = ""
        if name:
            return name[:24]
    sender = getattr(getattr(event, "message_obj", None), "sender", None)
    for attr in ("nickname", "card", "name", "user_id"):
        name = str(getattr(sender, attr, "") or "").strip()
        if name:
            return name[:24]
    return ""


class LlmemoryPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None):
        super().__init__(context)
        self.raw_config = dict(config) if config else {}
        _dd = str(self.raw_config.get("data_dir") or "").strip()
        if _dd:
            os.environ["LLMEMORY_DATA_DIR"] = _dd
        # 六维状态也挪到同一个数据目录，别一半在老地方一半在新地方。
        if not str(self.raw_config.get("adv_state_file") or "").strip():
            self.raw_config["adv_state_file"] = os.path.join(_data_dir(), "neurostate.json")
        self.cfg = load_config(self.raw_config, base_dir=_PLUGIN_ROOT)
        self.state = self.cfg.load_state()
        # 心情跟六维存一处，不另找目录。
        self.mood_path = os.path.join(
            os.path.dirname(self.cfg.state_file) or _data_dir(),
            "状态", "llmemory_mood.json")
        # 节点库跟碎片都跟着插件自己的数据目录，不猜外面的路径。
        self.nodes_path = os.path.join(_data_dir(), "nodes.json")
        self.book = load_book(self.nodes_path)
        # 取回那一套：模型跟着插件走，图跟节点库放一处；编码器懒加载，起服务不卡。
        self._vec_dir = os.path.join(_PLUGIN_ROOT, "models")
        self.encoder = Encoder(self._vec_dir)
        # 模型九十多兆，不随包走；缺就后台下一份。下不到也不挡事，检索退成时间窗。
        fetch_in_background(self._vec_dir, _vec_urls(self.raw_config),
                            sha=_vec_sha(self.raw_config),
                            on_done=self._on_vec_ready)
        self.graph = Graph(os.path.join(_data_dir(), "graph.npz"))
        self.recall = Recall(self.book, graph=self.graph, enc=self.encoder)
        self._recall_task = None
        self.last_umo = ""
        # 人格页和日记：目录走 P.data_dir()，换了路径就换一个库。
        wdir = str(self.raw_config.get("writer_dir") or "").strip() \
            or os.path.join(_data_dir(), "memory")
        self.writer = Writer(wdir, config=self.raw_config)
        try:                              # 还没人格页就先铺一版白板，别让注入那块永远空着
            seeded = self.writer.seed()
            if seeded:
                logger.info(f"[llmemory] 白板人格铺好了：{seeded}")
        except Exception as e:
            logger.warning(f"[llmemory] 白板人格没铺成：{e}")
        self._writer_task = None
        self._decay_task = None
        self.decay_file = os.path.join(_data_dir(), "状态", "decay.json")
        # 反思：只记看到哪一颗，跟六维、节点库分开存。
        self.reflector = Reflector(os.path.join(_data_dir(), "状态", "reflect.json"))
        self._reflect_task = None
        self.waker = Waker(
            self, nodes_path=self.nodes_path,
            muse_path=P.under("暗处", "muse.jsonl"))
        logger.info(f"[llmemory] 起来了，judge_mode={self.cfg.judge_mode}，"
                    f"state={self.cfg.state_file}，节点 {len(self.book)} 颗")
        _vec_err = deps_error()
        if _vec_err:
            logger.warning(f"[llmemory] 向量依赖缺着，检索走时间窗：{_vec_err}")

    def _on_vec_ready(self):
        """模型下齐了。编码器是懒加载的，下次要向量时自己会读到。"""
        logger.info("[llmemory] 向量模型到齐，检索那层能用了。")

    # ---- 判 ------------------------------------------------------------

    def _judge(self, text: str):
        """一句话 → (event, scale) 或 None。判不出就不动，不猜。"""
        if not text:
            return None
        mode = self.cfg.judge_mode
        if mode == "off":
            return None                    # 面板写了不判，就别偷偷判
        if mode == "model":
            try:
                got = judge_model(text, self.cfg)
            except Exception as e:
                logger.warning(f"[llmemory] 模型短判失败：{e}")
                got = None
            if got:
                return got                 # 模型判出来了就用它
        try:
            return judge_tuple(text)
        except Exception as e:            # 词表炸了也不该拦住对话
            logger.warning(f"[llmemory] 规则判失败：{e}")
            return None

    def _step(self, payload):
        """判 + 落状态 + 存盘。同步，调用方丢线程里跑。判出来的那一下要带回给提示用。"""
        raw, who = payload if isinstance(payload, tuple) else (payload, "")
        text = str(raw or "")
        who = str(who or "")
        self._note_live(who, text)
        judgement = self._judge(text)
        self.state = apply_judgement(self.state, judgement, **self.cfg.judge_kwargs())
        self.cfg.save_state(self.state)
        line = write_mood(self.state, judgement, self.mood_path)
        self._add_node(judgement, line, raw=text[:200], who=who)
        return judgement

    def _add_node(self, judgement, line: str, raw: str = "", who: str = "") -> None:
        """这一下也落一颗，神游才有地方走。落不上不算事。"""
        if not judgement or not line:
            return
        try:
            node = self.book.add(str(judgement[0]), float(judgement[1]),
                                 line, raw=raw, who=who)
        except Exception as e:
            logger.warning(f"[llmemory] 节点没落上：{e}")
            return
        try:
            self.recall.add(node)
        except Exception as e:                 # 连不上边不影响节点已经落了
            logger.warning(f"[llmemory] 向量没连上：{e}")

    def _note_session(self, event) -> None:
        """记下最近说话的地方，自唤醒醒了才知道往哪发。"""
        umo = getattr(event, "unified_msg_origin", "") or ""
        if umo and umo != self.last_umo:
            self.last_umo = umo
            write_target(umo)

    def _mood_line(self, judgement) -> str:
        """给这条消息配两截：这一下是什么，加上这一刻的反读。"""
        line = describe(self.state)
        if not line:
            return ""
        if not judgement:
            return line
        event_type, scale = judgement
        head = _EVENT_CN.get(event_type)
        if not head:
            return line
        if float(scale) >= 1.4:
            head += "，不轻"
        return f"{head}。{line}"

    # ---- 人格与日记 ----------------------------------------------------

    def _note_live(self, who: str, text: str) -> None:
        """把这一轮原话落到记忆库当天的实况里，写人格、日记时当素材。
        只落盘，不给对话加任何东西。"""
        text = (text or "").strip()
        if not text:
            return
        try:
            p = os.path.join(str(self.writer.mem), "live", self.writer.today + ".md")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write("%s：%s\n" % (who or "对方", text[:300].replace("\n", " ")))
        except Exception as e:
            logger.warning(f"[llmemory] 实况没落上：{e}")

    async def _gen_text(self, system: str, prompt: str) -> str:
        """调模型。配了 model_url + model_name 就直连那家；其次 writer_provider；
        都没有就走最近那个会话在用的那家。"""
        url = str(self.raw_config.get("model_url") or "").strip()
        mname = str(self.raw_config.get("model_name") or "").strip()
        mkey = str(self.raw_config.get("model_key") or "").strip()
        if url and mname:
            try:
                text = await asyncio.to_thread(
                    chat, url, mkey, mname,
                    [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
                    2000, 180.0, 0.8)
            except Exception as e:
                logger.warning(f"[llmemory] 直连模型失败：{e}")
                text = ""
            if text:
                return text
            logger.warning("[llmemory] 直连模型没回，退回 AstrBot 的 provider")
        pid = str(self.raw_config.get("writer_provider") or "").strip()
        try:
            if pid:
                return await self.context.llm_generate(
                    chat_provider_id=pid, prompt=prompt, system_prompt=system)
            prov = self.context.get_using_provider(self.last_umo) if self.last_umo else None
            if prov is None:
                return ""
            resp = await prov.text_chat(prompt=prompt, system_prompt=system)
            return getattr(resp, "completion_text", "") or ""
        except Exception as e:
            logger.warning(f"[llmemory] 调模型写东西失败：{e}")
            return ""

    async def _write_kind(self, kind: str) -> str:
        """写一份。到点和命令两处共用。返回落盘路径，没写成给空。"""
        label = "人格" if kind == "persona" else "日记"
        system, body = await asyncio.to_thread(self.writer.prompt, kind)
        text = await self._gen_text(system, body)
        if not text:
            self.writer.log("%s：模型没回，这一份空着" % label)
            return ""
        p = await asyncio.to_thread(self.writer.save, kind, text)
        if p:
            logger.info(f"[llmemory] 写下{label}：{p}")
        return p

    async def _writer_loop(self) -> None:
        """一分钟看一次，到点就写。写不写得成都不吵。"""
        while True:
            try:
                for kind in ("persona", "profile", "diary"):
                    if self.writer.due(kind):
                        await self._write_kind(kind)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[llmemory] 写人格/日记这一轮出错：{e}")
            await asyncio.sleep(60)

    def _writer_start(self) -> None:
        if self._writer_task is not None and not self._writer_task.done():
            return
        if not (self.writer.enabled("persona") or self.writer.enabled("profile")
                or self.writer.enabled("diary")):
            return
        self._writer_task = asyncio.create_task(self._writer_loop())
        logger.info(f"[llmemory] 写人格与日记的循环起了，库在 {self.writer.mem}")

    async def _writer_stop(self) -> None:
        task, self._writer_task = self._writer_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ---- 衰减 ----------------------------------------------------------

    def _decay_once(self) -> int:
        """把边按时间压一遍，压完存盘。同步，调用方丢线程里跑。"""
        if not len(self.graph.ids):
            return 0                       # 图还没读进来，不拿空的去盖盘
        days = float(self.raw_config.get("decay_tau_days") or 30.0)
        now = time.time()
        since = None
        try:
            with open(self.decay_file, encoding="utf-8") as f:
                since = float(json.load(f).get("ts") or 0.0) or None
        except Exception:
            since = None                   # 头一回，按节点自己的岁数算
        if since and since >= now:
            since = None                   # 钟回拨过，别压成负数
        changed = self.graph.decay(tau=max(days, 0.1) * 86400.0,
                                   now=now, since=since)
        # 先记下这一趟，再存图：存图失败也不至于下一趟把两回的量一起压
        try:
            os.makedirs(os.path.dirname(self.decay_file), exist_ok=True)
            with open(self.decay_file, "w", encoding="utf-8") as f:
                json.dump({"ts": now}, f)
        except Exception as e:
            logger.warning(f"[llmemory] 压边的时间戳没记上：{e}")
        if changed:
            try:
                self.graph.save()
            except Exception as e:
                logger.warning(f"[llmemory] 压完的图没存上：{e}")
        return changed

    async def _decay_loop(self) -> None:
        """隔几个钟头压一次。头一轮也等满一个间隔，别抢在取回把图读进来之前。"""
        hours = float(self.raw_config.get("decay_every_hours") or 4)
        gap = max(0.25, hours) * 3600.0
        while True:
            await asyncio.sleep(gap)
            try:
                n = await eventloop_run(self._decay_once)
                if n:
                    logger.info(f"[llmemory] 边压了一轮，动了 {n} 条。")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[llmemory] 压边这一轮出错：{e}")

    def _decay_start(self) -> None:
        if self._decay_task is not None and not self._decay_task.done():
            return
        if not self.raw_config.get("decay_enable", True):
            return
        self._decay_task = asyncio.create_task(self._decay_loop())
        logger.info("[llmemory] 边的衰减循环起了。")

    async def _decay_stop(self) -> None:
        task, self._decay_task = self._decay_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ---- 反思 ----------------------------------------------------------

    def _reflect_call(self):
        """读回话的手：借判用的那个模型，不另配。没配就返回 None。"""
        url, key, name = self.cfg.model_url, self.cfg.model_key, self.cfg.model_name
        if not (url and name):
            return None
        return lambda messages: chat(url, key, name, messages, 400, 120.0, 0.3)

    async def _reflect_once(self) -> int:
        """看一批。落了回一句的条数，没落回零。"""
        call = self._reflect_call()
        if call is None:
            return 0
        return await eventloop_run(self.reflector.run, self.book, call, self.recall.add)

    async def _reflect_loop(self) -> None:
        """隔几个钟头回看一批。头一轮也等满间隔，不跟取回暖机抢。"""
        hours = float(self.raw_config.get("reflect_every_hours") or 6)
        gap = max(0.5, hours) * 3600.0
        while True:
            await asyncio.sleep(gap)
            try:
                n = await self._reflect_once()
                if n:
                    logger.info("[llmemory] 反思落了一句。")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[llmemory] 反思这一轮出错：{e}")

    def _reflect_start(self) -> None:
        if self._reflect_task is not None and not self._reflect_task.done():
            return
        if not self.raw_config.get("reflect_enable", False):
            return
        if self._reflect_call() is None:
            logger.info("[llmemory] 反思没起：模型三样没配齐。")
            return
        self._reflect_task = asyncio.create_task(self._reflect_loop())
        logger.info("[llmemory] 反思的循环起了。")

    async def _reflect_stop(self) -> None:
        task, self._reflect_task = self._reflect_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ---- 钩子 ----------------------------------------------------------

    async def initialize(self) -> None:
        """自唤醒和神游都靠这一个循环。没开就干脆不起。"""
        self._recall_start()
        if getattr(self.waker, "drift_enable", False) or getattr(self.waker, "wake_enable", False):
            self.waker.start()
            logger.info("[llmemory] 神游与自唤醒的循环起了。")
        self._writer_start()
        self._decay_start()
        self._reflect_start()

    def _recall_start(self) -> None:
        """开模型、补历史向量、把边对齐，全是慢活儿，丢线程里跑。"""
        async def run():
            try:
                await eventloop_run(self._recall_warm)
            except Exception as e:
                logger.warning(f"[llmemory] 取回没热起来：{e}")
        try:
            self._recall_task = asyncio.get_event_loop().create_task(run())
        except Exception as e:
            logger.warning(f"[llmemory] 预热任务没起：{e}")

    def _recall_warm(self) -> None:
        """线程里干：载模型 -> 读图 -> 补缺的向量 -> 边不对就重算一次。"""
        if not self.encoder.ready:
            logger.warning("[llmemory] 没找着向量模型，取回先退成时间窗。")
            return
        self.encoder.load()
        self.graph.load()
        added = self.recall.backfill()
        edges = len(self.graph.edges)
        if added or edges == 0:
            edges = self.recall.rebuild()
        logger.info(f"[llmemory] 取回醒了：图 {len(self.graph.ids)} 行，"
                    f"补 {added} 条向量，{edges} 条边。")

    @filter.on_llm_request()
    async def _on_request(self, event: AstrMessageEvent, req):
        self._note_session(event)
        if not self.cfg.enable:
            return
        say = event.message_str or ""
        try:
            judgement = await eventloop_run(self._step, (say, _speaker(event)))
        except Exception as e:
            logger.warning(f"[llmemory] 状态更新失败：{e}")
            judgement = None
        blocks = []
        # 人格页：整页带上，卡 writer 的字数上限（默认 900），放最前
        try:
            page = _read(self.writer.persona, self.writer.chars("persona"))
        except Exception as e:
            logger.warning(f"[llmemory] 人格页读不出来：{e}")
            page = ""
        if page:
            blocks.append("【我是谁】\n" + page)
        # 他那一页：结构是我定的那几节，模型写得出就写，写不出就空着
        try:
            prof = _read(self.writer.target("profile"), self.writer.chars("profile"))
        except Exception as e:
            logger.warning(f"[llmemory] 画像读不出来：{e}")
            prof = ""
        if prof:
            blocks.append("【关于他】\n" + prof)
        # 取回独立成段，不受心情开关牵连
        try:
            items = await eventloop_run(self.recall.query, say)
        except Exception as e:
            logger.warning(f"[llmemory] 想不起来：{e}")
            items = []
        if items:
            try:
                block = self.recall.render(items)
            except Exception as e:
                logger.warning(f"[llmemory] 想起的拼不出来：{e}")
                block = ""
            if block:
                blocks.append(block)
        if self.cfg.affect_tone:
            try:
                line = self._mood_line(judgement)
            except Exception as e:
                logger.warning(f"[llmemory] 心情拼不出来：{e}")
                line = ""
            if line:
                blocks.append(line)
        if blocks:
            # 人格页、画像、想起的、心情句都是这一轮临时给的：走 extra_user_content_parts
            # 并标 temp，只喂给模型，不进会话历史，也不动 system_prompt（缓存命中靠它）。
            try:
                for block in blocks:
                    req.extra_user_content_parts.append(
                        TextPart(text=block).mark_as_temp())
            except Exception as e:
                logger.warning(f"[llmemory] 注入没挂上：{e}")

    @filter.command("情绪")
    async def show_state(self, event: AstrMessageEvent):
        """看一眼当前六维。调试用。"""
        yield event.plain_result(describe(self.state) or "(空)")

    @filter.command("写日记")
    async def write_diary_now(self, event: AstrMessageEvent):
        """现在就写今天的日记。"""
        p = await self._write_kind("diary")
        yield event.plain_result(("写好了：" + os.path.basename(str(p))) if p else "没写成，模型没回或写不进。")
        yield event.plain_result(self.writer.status())

    @filter.command("写人格")
    async def write_persona_now(self, event: AstrMessageEvent):
        """现在就重写一版人格页。"""
        p = await self._write_kind("persona")
        yield event.plain_result(("写好了：" + os.path.basename(str(p))) if p else "没写成，模型没回或写不进。")
        yield event.plain_result(self.writer.status())

    @filter.command("写画像")
    async def write_profile_now(self, event: AstrMessageEvent):
        """现在就从原话重蒸一遍他那一页。"""
        p = await self._write_kind("profile")
        yield event.plain_result(("写好了：" + os.path.basename(str(p))) if p else "没写成，模型没回或写不进。")
        yield event.plain_result(self.writer.status())

    async def terminate(self) -> None:
        try:
            await self.waker.stop()
        except Exception as e:
            logger.warning(f"[llmemory] 自唤醒没停干净：{e}")
        await self._writer_stop()
        await self._decay_stop()
        await self._reflect_stop()
        self.cfg.save_state(self.state)
        logger.info("[llmemory] 关了，状态已落盘。")


async def eventloop_run(fn, *args):
    """把同步的活儿丢线程，别堵事件循环。"""
    import asyncio
    return await asyncio.to_thread(fn, *args)
