"""llmemory 插件入口。

T0 只做一件事的闭环：收一句话 → 判 → 落到六维 → 存盘。
注入那层只给一句，不抄日志，也不替模型组织措辞。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:                                      # 按包加载（AstrBot 走这条）
    from .memorydata.affect import (
        apply_judgement, chat, describe, judge_model, judge_tuple, load_config,
    )
    from .memorydata.graph import Graph
    from .memorydata.muse import DEFAULT_PATH as MUSE_PATH
    from .memorydata.nodes import load_book
    from .memorydata.recall import Recall
    from .memorydata.reflect import Reflector
    from .memorydata.vec import Encoder, deps_error
    from .memorydata.model_fetch import fetch_in_background
    from .memorydata import paths as _paths
    from .memorydata.wake import Waker, write_target
    from .memorydata.writer import Writer, _read
except ImportError:                       # 插件目录在 sys.path 上时
    from memorydata.affect import (
        apply_judgement, chat, describe, judge_model, judge_tuple, load_config,
    )
    from memorydata.graph import Graph
    from memorydata.muse import DEFAULT_PATH as MUSE_PATH
    from memorydata.nodes import load_book
    from memorydata.recall import Recall
    from memorydata.reflect import Reflector
    from memorydata.vec import Encoder, deps_error
    from memorydata.model_fetch import fetch_in_background
    from memorydata import paths as _paths
    from memorydata.wake import Waker, write_target
    from memorydata.writer import Writer, _read

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
    """插件的数据目录，绝对路径。默认走框架给的 data/plugin_data/<插件名>，
    升级插件不碰它；配置了 data_dir 或环境变量 LLMEMORY_DATA_DIR 就挪过去，
    那是显式挪库的口子。"""
    return _paths.data_dir()


MOOD_OUT = os.path.join(_data_dir(), "状态", "llmemory_mood.json")

# 判不出情绪的那句，也不是没发生。落一颗「日常话」节点：只进图，六维一分不动。
TALK_EVENT = "talk"
TALK_MIN_CHARS = 6      # 去标点后不到六个字，当语气，不当一句
TALK_MAX = 800       # 一颗是一轮：他那句加我这句，别把回话切掉
DEDUP_GAP = 60.0     # 同一句在这个窗口里只认一次，挡 agent 流反复请求
PARK_TTL = 180.0     # 挂了这么久还没等到回话，就照他一个人那句落掉


def _worth_talk(text: str) -> bool:
    """够一句日常话才落：不是纯语气、不是光一条链接、不是一串符号。"""
    t = (text or "").strip()
    if not t or t.startswith(("http://", "https://")):
        return False
    return len(re.sub(r"[\s\W_]+", "", t, flags=re.UNICODE)) >= TALK_MIN_CHARS


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
        logger.warning(f"[llmemory] 情绪状态写入失败：{e}")
        return ""
    return line


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


def _reply_text(resp) -> str:
    """这条回复真正说出口的那段。工具调用、流式切片、空话都算没说。"""
    if getattr(resp, "role", "") not in ("assistant", ""):
        return ""
    if getattr(resp, "is_chunk", False):
        return ""
    if getattr(resp, "tools_call_name", None):
        return ""
    text = ""
    chain = getattr(resp, "result_chain", None)
    if chain is not None:
        try:
            text = chain.get_plain_text()
        except Exception:
            text = ""
    if not str(text).strip():
        text = getattr(resp, "completion_text", "") or ""
    return str(text).strip()


@register(
    PLUGIN_NAME,
    "",
    "情绪与内分泌层：把一句话判成五个事件之一，落到六维状态上，跨轮存盘。不接模型也能跑。",
    "0.1.8",
    "",
)
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
        # 每个人最近落的那颗日常话，一分钟内的话并进去
        # 谁最近说了哪句：agent 流一轮请求好几回，同一句别反复记
        self._seen = {}
        # 一轮问答挂在这儿等回话：他这句先判，节点等我这句到了再落
        self._pending = {}
        # 人格页和日记：目录跟着插件自己的 plugin_data 走，换了路径就换一个库。
        wdir = str(self.raw_config.get("writer_dir") or "").strip() \
            or os.path.join(_data_dir(), "memory")
        self.writer = Writer(wdir, config=self.raw_config)
        self._writer_task = None
        self._decay_task = None
        self.decay_file = os.path.join(_data_dir(), "状态", "decay.json")
        # 反思：只记看到哪一颗，跟六维、节点库分开存。
        self.reflector = Reflector(os.path.join(_data_dir(), "状态", "reflect.json"))
        self._reflect_task = None
        self.waker = Waker(
            self, nodes_path=self.nodes_path,
            muse_path=os.path.join(_data_dir(), "暗处", "muse.jsonl"))
        logger.info(f"[llmemory] 初始化完成，judge_mode={self.cfg.judge_mode}，"
                    f"state={self.cfg.state_file}，节点数 {len(self.book)}")
        _vec_err = deps_error()
        if _vec_err:
            logger.warning(f"[llmemory] 向量依赖缺失，检索降级为时间窗：{_vec_err}")

    def _on_vec_ready(self):
        """模型下齐了。编码器是懒加载的，下次要向量时自己会读到。"""
        logger.info("[llmemory] 向量模型已就绪，向量检索可用。")

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
                logger.warning(f"[llmemory] 模型判官调用失败：{e}")
                got = None
            if got:
                return got                 # 模型判出来了就用它
        try:
            return judge_tuple(text)
        except Exception as e:            # 词表炸了也不该拦住对话
            logger.warning(f"[llmemory] 规则判官调用失败：{e}")
            return None

    def _step(self, payload):
        """判 + 落状态 + 存盘，把这一轮挂起来等回话。同步，调用方丢线程里跑。

        一颗节点是一轮问答：他这句先判，六维当场动（这一轮拼提示要用），
        节点不在这儿落，等我这句回来，两半并一颗。
        """
        text, who, umo = payload if isinstance(payload, tuple) else (payload, "", "")
        text, who, umo = str(text or ""), str(who or ""), str(umo or "")
        self._note_live(who, text)
        # 一句话只认一次。agent 流一轮里模型会请求好几回，
        # message_str 反复是同一句，不去重就一层层往库里叠。
        hit = self._seen.get(who)
        fresh = not (hit and hit[0] == text and 0.0 <= time.time() - hit[1] <= DEDUP_GAP)
        if fresh:
            judgement = self._judge(text)
            self.state = apply_judgement(self.state, judgement, **self.cfg.judge_kwargs())
            self.cfg.save_state(self.state)
            line = write_mood(self.state, judgement, self.mood_path)
        else:
            judgement, line = hit[2], hit[3]
        self._park(umo, text, who, judgement, line)
        if fresh and text:
            self._seen[who] = (text, time.time(), judgement, line)
            if len(self._seen) > 256:
                now = time.time()
                self._seen = {k: v for k, v in self._seen.items() if now - v[1] <= DEDUP_GAP}
        return judgement

    def _park(self, umo: str, text: str, who: str, judgement, line: str) -> None:
        """这一轮先挂在这儿等回话。上一轮要是没等到回话，先照他那句落掉，别一直压着。"""
        if not line:
            return
        old = self._pending.get(umo)
        if old is not None and time.time() - old[4] > PARK_TTL:
            self._pending.pop(umo, None)
            self._land(old, "")
        self._pending[umo] = (text, who, judgement, line, time.time())

    def _land(self, pend, say: str) -> None:
        """一轮收口：他那句在前，我这句在后，一颗落下去。"""
        text, who, judgement, line = pend[0], pend[1], pend[2], pend[3]
        his, mine = (text or "").strip(), (say or "").strip()
        raw = f"他：{his}\n我：{mine}" if mine else f"他：{his}"
        self._add_node(judgement, line, raw=raw[-TALK_MAX:], who=who, his=his)

    def _add_node(self, judgement, line: str, raw: str = "", who: str = "",
                  his: str = "") -> None:
        """一轮落一颗，神游才有地方走。落不上不算事。

        判出情绪就落情绪那颗；判不出也落，记成日常话。
        只留下被戳的那几下，热闹的话全漏掉，库就不像日子了。
        算不算数看他那句，别拿我这句的篇幅替他凑。
        """
        if not line:
            return
        if judgement:
            event, scale = str(judgement[0]), float(judgement[1])
        else:
            if not _worth_talk(his or raw):
                return
            event, scale = TALK_EVENT, 0.0
        try:
            node = self.book.add(event, scale, line, raw=raw, who=who)
        except Exception as e:
            logger.warning(f"[llmemory] 节点写入失败：{e}")
            return
        try:
            self.recall.add(node)
        except Exception as e:                 # 连不上边不影响节点已经落了
            logger.warning(f"[llmemory] 向量生成失败：{e}")

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
            logger.warning(f"[llmemory] 实况写入失败：{e}")

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
                logger.warning(f"[llmemory] 直连模型调用失败：{e}")
                text = ""
            if text:
                return text
            logger.warning("[llmemory] 直连模型未返回，回退至 AstrBot provider")
        pid = str(self.raw_config.get("writer_provider") or "").strip()
        try:
            if pid:
                return await self.context.llm_generate(
                    chat_provider_id=pid, prompt=prompt, system_prompt=system)
            prov = self.context.get_using_provider(self.last_umo or None)
            if prov is None:
                return ""
            resp = await prov.text_chat(prompt=prompt, system_prompt=system)
            return getattr(resp, "completion_text", "") or ""
        except Exception as e:
            logger.warning(f"[llmemory] 调用模型生成内容失败：{e}")
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
            logger.info(f"[llmemory] 已生成{label}：{p}")
        return p

    async def _writer_loop(self) -> None:
        """一分钟看一次，到点就写。写不写得成都不吵。"""
        while True:
            try:
                for kind in ("persona", "diary"):
                    if self.writer.due(kind):
                        await self._write_kind(kind)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[llmemory] 人格/日记生成本轮出错：{e}")
            await asyncio.sleep(60)

    def _writer_start(self) -> None:
        if self._writer_task is not None and not self._writer_task.done():
            return
        if not (self.writer.enabled("persona") or self.writer.enabled("diary")):
            return
        self._writer_task = asyncio.create_task(self._writer_loop())
        logger.info(f"[llmemory] 人格与日记定时任务已启动，数据目录：{self.writer.mem}")

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
        """把两套边按时间压一遍，压完存盘。同步，调用方丢线程里跑。"""
        days = float(self.raw_config.get("decay_tau_days") or 30.0)
        tau = max(days, 0.1) * 86400.0
        now = time.time()
        since = None
        try:
            with open(self.decay_file, encoding="utf-8") as f:
                since = float(json.load(f).get("ts") or 0.0) or None
        except Exception:
            since = None                   # 头一回，按节点自己的岁数算
        if since and since >= now:
            since = None                   # 钟回拨过，别压成负数
        changed = 0
        if len(self.graph.ids):
            changed = self.graph.decay(tau=tau, now=now, since=since)
        else:
            logger.warning("[llmemory] 图未加载，本轮仅衰减节点库边权。")
        # 节点库那套边也压：神游走的就是这张图，不压会越抽越粗
        try:
            changed += self.book.decay(tau=tau, now=now, since=since)
        except Exception as e:
            logger.warning(f"[llmemory] 节点边衰减失败：{e}")
        # 先记下这一趟，再存图：存图失败也不至于下一趟把两回的量一起压
        try:
            os.makedirs(os.path.dirname(self.decay_file), exist_ok=True)
            with open(self.decay_file, "w", encoding="utf-8") as f:
                json.dump({"ts": now}, f)
        except Exception as e:
            logger.warning(f"[llmemory] 衰减时间戳写入失败：{e}")
        if changed and len(self.graph.ids):
            try:
                self.graph.save()
            except Exception as e:
                logger.warning(f"[llmemory] 衰减后的图保存失败：{e}")
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
                    logger.info(f"[llmemory] 边权衰减完成，变更 {n} 条。")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[llmemory] 边权衰减本轮出错：{e}")

    def _decay_start(self) -> None:
        if self._decay_task is not None and not self._decay_task.done():
            return
        if not self.raw_config.get("decay_enable", True):
            return
        self._decay_task = asyncio.create_task(self._decay_loop())
        logger.info("[llmemory] 边权衰减定时任务已启动。")

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
                    logger.info("[llmemory] 反思生成完成。")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[llmemory] 反思本轮出错：{e}")

    def _reflect_start(self) -> None:
        if self._reflect_task is not None and not self._reflect_task.done():
            return
        if not self.raw_config.get("reflect_enable", False):
            return
        if self._reflect_call() is None:
            logger.info("[llmemory] 反思未启动：模型配置不完整。")
            return
        self._reflect_task = asyncio.create_task(self._reflect_loop())
        logger.info("[llmemory] 反思定时任务已启动。")

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
            logger.info("[llmemory] 神游与自唤醒定时任务已启动。")
        self._writer_start()
        self._decay_start()
        self._reflect_start()

    def _recall_start(self) -> None:
        """开模型、补历史向量、把边对齐，全是慢活儿，丢线程里跑。"""
        async def run():
            try:
                await eventloop_run(self._recall_warm)
            except Exception as e:
                logger.warning(f"[llmemory] 检索预热失败：{e}")
        try:
            self._recall_task = asyncio.get_event_loop().create_task(run())
        except Exception as e:
            logger.warning(f"[llmemory] 预热任务创建失败：{e}")

    def _recall_warm(self) -> None:
        """线程里干：载模型 -> 读图 -> 补缺的向量 -> 边不对就重算一次。"""
        if not self.encoder.ready:
            logger.warning("[llmemory] 未找到向量模型，检索降级为时间窗。")
            return
        self.encoder.load()
        self.graph.load()
        added = self.recall.backfill()
        edges = len(self.graph.edges)
        if added or edges == 0:
            edges = self.recall.rebuild()
        logger.info(f"[llmemory] 检索预热完成：图 {len(self.graph.ids)} 行，"
                    f"新增向量 {added} 条，边 {edges} 条。")

    @filter.on_llm_request()
    async def _on_request(self, event: AstrMessageEvent, req):
        self._note_session(event)
        if not self.cfg.enable:
            return
        say = event.message_str or ""
        try:
            judgement = await eventloop_run(
                self._step, (say, _speaker(event), event.unified_msg_origin or ""))
        except Exception as e:
            logger.warning(f"[llmemory] 状态更新失败：{e}")
            judgement = None
        blocks = []
        # 人格页：整页带上，卡 writer 的字数上限（默认 900），接在平台人设后面当补充
        try:
            page = _read(self.writer.persona, self.writer.chars("persona"))
        except Exception as e:
            logger.warning(f"[llmemory] 人格页读取失败：{e}")
            page = ""
        # 白板稿不注入：名字还没定、话还没长开的那种，别去压平台自己写的人设
        if page and any(w in page for w in ("还没定", "名字还没定", "名字等有人给我起")):
            logger.info("[llmemory] 人格页为空，本轮跳过注入。")
            page = ""
        if page:
            blocks.append("【我是谁·相处里攒出来的一页】\n"
                          "这一页只当补充。平台/system 里已经定好的人设优先，"
                          "口气、称呼、来路都照那份来；两处不一致，以平台那份为准。\n" + page)
        # 取回独立成段，不受心情开关牵连
        try:
            items = await eventloop_run(self.recall.query, say)
        except Exception as e:
            logger.warning(f"[llmemory] 记忆检索失败：{e}")
            items = []
        if items:
            try:
                block = self.recall.render(items)
            except Exception as e:
                logger.warning(f"[llmemory] 检索结果组装失败：{e}")
                block = ""
            if block:
                blocks.append(block)
        if self.cfg.affect_tone:
            try:
                line = self._mood_line(judgement)
            except Exception as e:
                logger.warning(f"[llmemory] 情绪注入文本生成失败：{e}")
                line = ""
            if line:
                blocks.append(line)
        if blocks:
            req.system_prompt = (req.system_prompt or "") + "\n" + "\n".join(blocks)

    @filter.on_llm_response()
    async def _on_response(self, event: AstrMessageEvent, resp):
        """一轮问答收口：我这句出口了，和他那句并成一颗落。

        生成失败、被工具调用截断、只有流式切片，都当没说话，这一轮继续挂着。
        """
        try:
            umo = getattr(event, "unified_msg_origin", "") or ""
            say = _reply_text(resp)
            if not say or not umo:
                return
            pend = self._pending.pop(umo, None)
            if pend is None:
                return
            try:
                await eventloop_run(self._land, pend, say)
            except BaseException:
                self._pending.setdefault(umo, pend)   # 没落成就放回去，等下一轮兜底
                raise
        except Exception as e:
            logger.warning(f"[llmemory] 本轮状态写入失败：{e}")

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

    async def terminate(self) -> None:
        try:
            await self.waker.stop()
        except Exception as e:
            logger.warning(f"[llmemory] 自唤醒停止异常：{e}")
        await self._writer_stop()
        await self._decay_stop()
        await self._reflect_stop()
        self.cfg.save_state(self.state)
        logger.info("[llmemory] 插件已卸载，状态已保存。")


async def eventloop_run(fn, *args):
    """把同步的活儿丢线程，别堵事件循环。"""
    import asyncio
    return await asyncio.to_thread(fn, *args)
