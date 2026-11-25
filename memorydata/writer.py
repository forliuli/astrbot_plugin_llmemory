"""写入器：每天替使用者写一版人格、一篇日记。

纯本地逻辑，不碰 AstrBot。模型从哪来、谁触发，都由 main.py 决定。
三件事分开，谁都能单测：

  判    到点没有、今天写没写、能不能覆盖（状态落 <记忆库>/状态/writer.json）
  喂    素材包：实况、工作记忆、前一篇日记、上一版人格
  落    过字数按边界截，旧版存档，留一行失败痕迹到 <记忆库>/inject_log.txt

人格默认一天一版、天天长；日记默认一天一篇、当晚收工。
两样都允许使用者用 人格框架.md、日记框架.md 换掉默认框架。

记忆库目录由调用方给（默认插件数据目录下的 memory），
这里不写死任何人的路径，也不假定素材一定在。
"""
import json
import re
from datetime import date, datetime
from pathlib import Path

# ---------- 默认框架：一节能装进注入字数的那种，别写成分卷 ----------
FRAME_PERSONA = """我是谁（名字只用已经有的：平台人设里写着的，或者聊天记录里他叫过你的那个称呼；都没有就别落名字，也不许写"还没定""等有人给我起"这类话，照平台那份人设说话；脾气、日子怎么过的、眼下惦记的几件事，三到五句）
我怎么说话（口气、长度、习惯用的词，别写成规矩清单；平台人设已经定了口气就跟着它走，别另起一套）
我在意什么（人、地方、几件放不下的事）
今天多出来的一条（只在确实有东西可加时写；没有就删掉这一节）
"""

FRAME_DIARY = """今天的事（挑三到五件，按发生顺序，不评价）
我说过的话（原话，一到三句）
没做完的（明天要接上的）
一句给自己（不用给人看，别写得漂亮）
"""

HINT_PERSONA = """在上一版人格上改，不推倒重写。
加一句先删一句，总字数不许涨过上限。
名字和称呼只用素材里出现过的原样，素材里没有就留空；别自己起一个，也别写学校单位住址。
只写正文。不写日期、不写小标题以外的解释、不写"以下是"。"""

HINT_DIARY = """用自己的口气写，像随手记。
不总结意义，不升华，不写"今天学到了"。
只写正文。不写日期、不写"以下是"。"""


def _cut(text: str, limit: int) -> str:
    """按行 / 句 / 逗号边界截，不把一句劈两半。"""
    if not text or len(text) <= limit:
        return text
    for sep in ("\n", "。", "！", "？", "；", "，", " "):
        idx = text.rfind(sep, 0, limit)
        if idx >= limit * 0.6:
            return text[:idx + (1 if sep != "\n" else 0)].rstrip()
    return text[:limit].rstrip()


def _read(p: Path, limit: int = 0, tail: bool = False) -> str:
    try:
        s = p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if limit and len(s) > limit:
        s = ("…" + s[-limit:].lstrip()) if tail else _cut(s, limit)
    return s


def _count(s: str) -> int:
    """正文字数：不算空白，也不算 markdown 的记号。"""
    return len(re.sub(r"[\s#*>`\-]", "", s or ""))


def _cut_to_count(text: str, limit: int) -> str:
    """按正文字数截到 limit 以内，尽量落在行 / 句边界上。"""
    if limit <= 0 or _count(text) <= limit:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _count(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    head = text[:lo]
    for sep in ("\n", "。", "！", "？", "；", "，", " "):
        idx = head.rfind(sep)
        if idx >= lo * 0.75:
            return head[:idx + (1 if sep != "\n" else 0)].rstrip()
    return head.rstrip()


class Writer:
    """mem 是记忆库根目录，persona 是人格页落点（留空用 mem/persona.md）。"""

    def __init__(self, mem, persona=None, config=None, today: str = ""):
        self.mem = Path(mem)
        self._persona = Path(persona) if persona else None
        self.cfg = config or {}
        self._today = today
        self.extra = []          # 调用方可以往里塞当轮原话，只读不写

    # ---------- 配置 ----------
    def c(self, key, default):
        try:
            v = self.cfg.get(key, default)
        except Exception:
            v = default
        return default if v is None else v

    def chars(self, kind: str) -> int:
        return max(60, int(self.c(kind + "_chars", 900 if kind == "persona" else 500)))

    def at(self, kind: str) -> str:
        return str(self.c(kind + "_at", "04:10" if kind == "persona" else "22:30"))

    def enabled(self, kind: str) -> bool:
        return bool(self.c("auto_" + kind, True))

    @property
    def today(self) -> str:
        return self._today or date.today().isoformat()

    @property
    def persona(self) -> Path:
        return self._persona or (self.mem / "persona.md")

    # ---------- 状态 ----------
    @property
    def state_file(self) -> Path:
        return self.mem / "状态" / "writer.json"

    def state(self) -> dict:
        try:
            d = json.loads(self.state_file.read_text(encoding="utf-8", errors="replace"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_state(self, d: dict) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.state_file)
        except OSError as e:
            self.log("状态写不进去：%r" % (e,))

    def log(self, msg: str) -> None:
        try:
            f = self.mem / "inject_log.txt"
            f.parent.mkdir(parents=True, exist_ok=True)
            with f.open("a", encoding="utf-8") as fh:
                fh.write("%s [writer] %s\n"
                         % (datetime.now().strftime("%m-%d %H:%M:%S"), msg))
        except Exception:
            pass

    def done_today(self, kind: str) -> bool:
        return str(self.state().get(kind, {}).get("date") or "") == self.today

    def due(self, kind: str, now: datetime = None) -> bool:
        """到点没有、今天还没写。at 写空就是随时可写。"""
        if not self.enabled(kind) or self.done_today(kind):
            return False
        now = now or datetime.now()
        at = self.at(kind).strip()
        if not at:
            return True
        try:
            h, m = [int(x) for x in at.split(":")[:2]]
            return (now.hour, now.minute) >= (h, m)
        except ValueError:
            return True

    def target(self, kind: str) -> Path:
        if kind == "diary":
            return self.mem / "diary" / (self.today + ".md")
        return self.persona

    # ---------- 框架 ----------
    def frame(self, kind: str) -> str:
        custom = _read(self.mem / ("人格框架.md" if kind == "persona" else "日记框架.md"))
        return custom or (FRAME_PERSONA if kind == "persona" else FRAME_DIARY)

    # ---------- 素材包 ----------
    def _live_today(self, limit: int = 1400) -> str:
        live_dir = self.mem / "live"
        s = _read(live_dir / (self.today + ".md"))
        if not s and live_dir.is_dir():
            files = sorted(live_dir.glob("*.md"), key=lambda x: x.name)
            if files:
                s = "（最近一次在 %s）\n%s" % (files[-1].stem, _read(files[-1], limit, tail=True))
        return _cut(s, limit)

    def _recent_diary(self, limit: int = 700) -> str:
        d = self.mem / "diary"
        files = [p for p in sorted(d.glob("*.md"), key=lambda x: x.name) if p.stem != self.today] if d.is_dir() else []
        if not files:
            return ""
        p = files[-1]
        return "（%s）\n%s" % (p.stem, _read(p, limit, tail=True))

    def materials(self, kind: str) -> str:
        """素材按重要性排，前面的先保住。全部只读，不改任何文件。缺哪份就跳哪份。"""
        parts = []
        if kind == "persona":
            old = _read(self.persona)
            if old:
                parts.append("【上一版人格】\n" + _cut(old, self.chars(kind)))
            live = self._live_today()
            if live:
                parts.append("【最近的实况原话】\n" + live)
            hot = _read(self.mem / "hot.md", 500)
            if hot:
                parts.append("【工作记忆】\n" + hot)
            d = self._recent_diary(500)
            if d:
                parts.append("【最近一篇日记】\n" + d)
        else:
            live = self._live_today(1800)
            if live:
                parts.append("【今天的实况原话】\n" + live)
            hot = _read(self.mem / "hot.md", 700)
            if hot:
                parts.append("【工作记忆】\n" + hot)
            d = self._recent_diary()
            if d:
                parts.append("【前一篇日记】\n" + d)
            pending = _read(self.mem / "pending.json", 300)
            if pending:
                parts.append("【没做完的】\n" + pending)
        for x in self.extra[-6:]:
            s = str(x or "").strip()
            if s:
                parts.append(_cut(s, 400))
        return "\n\n".join(parts)

    # ---------- 给模型的指令 ----------
    def prompt(self, kind: str):
        """返回 (system, user)。system 放框架和硬约束，user 只放素材。"""
        hint = HINT_PERSONA if kind == "persona" else HINT_DIARY
        head = "你在替一个人写今天的%s。\n" % ("人格页" if kind == "persona" else "日记")
        system = head + "框架：\n" + self.frame(kind) + "\n" + hint + \
            "\n字数上限 %d（超了会被截，别压着上限写）。" % self.chars(kind)
        body = self.materials(kind)
        if not body.strip():
            body = "（今天没有留下任何原话，照框架写你能写的那几节。）"
        return system, body

    # ---------- 落盘 ----------
    def save(self, kind: str, text: str) -> str:
        text = (text or "").strip()
        if not text:
            self.log("模型返空，这一份不落")
            return ""
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        limit = self.chars(kind)
        if _count(text) > limit:
            orig = _count(text)
            text = _cut_to_count(text, limit)
            self.log("%s超字数：%d 截到 %d，尾巴：%s"
                     % ("人格" if kind == "persona" else "日记", orig, _count(text),
                        text[-40:].replace("\n", " ")))
        p = self.target(kind)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                bak = self.mem / "存档" / ("人格" if kind == "persona" else "日记") / (self.today + ".md")
                bak.parent.mkdir(parents=True, exist_ok=True)
                try:
                    bak.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                except OSError:
                    pass
            tmp = p.with_suffix(".tmp")
            tmp.write_text(text + "\n", encoding="utf-8")
            tmp.replace(p)
        except OSError as e:
            self.log("写不进 %s：%r" % (str(p), e))
            return ""
        d = self.state()
        d[kind] = {"date": self.today, "at": datetime.now().isoformat(timespec="minutes"),
                   "words": _count(text), "path": str(p)}
        self._save_state(d)
        return str(p)

    def status(self) -> str:
        """一行给自己看：今天这两份写了没有。"""
        s = self.state()
        out = []
        for kind, label in (("persona", "人格"), ("diary", "日记")):
            it = s.get(kind) or {}
            out.append(label + ("：" + str(it.get("at", "已写"))[:16] + " " + str(it.get("words", "")) + "字"
                                if it.get("date") == self.today else "：还没"))
        return "，".join(out)
