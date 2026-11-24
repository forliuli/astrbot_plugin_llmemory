"""六维神经递质状态模型。

底子：github.com/kagioneko/neurostate-engine (MIT)
  - interaction_matrix.py 的 MATRIX_A / EXTERNAL_FORCE_COEFFS / clamp
  - update_engine.py 的 _EQUILIBRIUM / EVENT_POWER_MAP
系数一个没动。不搬的：它对外的自报日志、依赖类型诊断、那一套日文话术。

改了三处：
  1. 自然回落按真实时间 1-exp(-dt/τ)，不按步数拉 5%。
     按步数算，搁三天只走一步，等于没塌。
  2. 矩阵只留跨维耦合（对角交给 τ 衰减）。原实现直接拿 MATRIX_A 乘原值，
     行和 0.80~1.05 不等，均衡点本身不是不动点，空转也会漂走。
     现在作用在「相对基线的偏离量」上，基线是真不动点。
  3. corruption 自己一条上升路：S 连续掉、O 长期低位，按小时缓升；
     不再只等 D>90 那道常年摆设的闸。

状态量都在 [0, 100]。事件是脉冲，时间演化是连续函数，两件事分开调。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace

__all__ = [
    "DIMS",
    "DIM_NAMES",
    "EQUILIBRIUM",
    "EVENT_POWER",
    "TAU_SECONDS",
    "NeuroState",
    "new_state",
    "advance",
    "apply_event",
    "apply_judgement",
    "describe",
    "evaluate_gate",
    "parse_judgement",
]

# ---------------------------------------------------------------- 常量

DIMS = ("D", "S", "C", "O", "G", "E")

DIM_NAMES = {
    "D": "多巴胺（奖赏动机）",
    "S": "血清素（稳定）",
    "C": "乙酰胆碱（专注）",
    "O": "催产素（亲密）",
    "G": "GABA（放松抑制）",
    "E": "内啡肽（快感）",
}

# 安静时的落点。用引擎代码里的 _EQUILIBRIUM，spec 里写 D50/O0 那版作废。
EQUILIBRIUM = {"D": 40.0, "S": 50.0, "C": 50.0, "O": 20.0, "G": 50.0, "E": 40.0}

MATRIX_A = (
    (0.90, -0.10, 0.10, 0.00, -0.20, 0.10),  # D
    (-0.10, 0.90, -0.05, 0.00, 0.20, -0.05),  # S
    (0.10, 0.00, 0.95, 0.00, 0.00, 0.10),  # C
    (0.00, 0.20, 0.00, 0.90, -0.10, 0.20),  # O
    (-0.15, 0.30, 0.00, -0.10, 0.90, -0.20),  # G
    (0.20, -0.20, 0.10, 0.10, -0.30, 0.90),  # E
)

EXTERNAL_FORCE_COEFFS = (1.0, 0.0, 0.5, 0.3, -0.4, 0.6)

EVENT_POWER = {
    "praise": 2.0,      # 夸
    "criticism": -1.0,  # 批
    "bonding": 1.0,     # 贴
    "stress": 1.5,      # 压
    "relaxation": -0.5,  # 松
}

# S 不吃 EXTERNAL_FORCE_COEFFS。那张表里 S 是 0.0，结果批评/S 压不下去，
# 单列一张：负面事件压 S，正面事件抬 S。value 是 scale=1 时的绝对分。
S_FORCE = {
    "praise": 0.6,
    "criticism": -1.8,
    "bonding": 0.8,
    "stress": -1.8,
    "relaxation": 0.5,
}

# O 也单列一张。EXTERNAL_FORCE_COEFFS 里 O 只有 0.3：夸一句只抬 0.6 分，
# 离 0.05 的地板（1.25 分）差一截，亲近的劲儿全被吃掉，轻的事件一律读成「心里平」。
# 贴一下得真抬起来；批评、压不往亲密那头走，量就压得很低。
O_FORCE = {
    "praise": 1.5,
    "criticism": -1.5,
    "bonding": 3.5,
    "stress": -2.5,
    "relaxation": 0.5,
}

# 事件这一下带来的脏。跟漂移分开：漂移看「离基线多远」，事件看「这一下是什么」。
CORRUPTION_EVENT_DELTA = {
    "praise": -0.5,
    "criticism": 1.5,
    "bonding": -0.5,
    "stress": 1.0,
    "relaxation": -1.5,
}

# 各自回落到基线的半程时间。T2 只定了 D/S/O，其余先给保守值，一周后按曲线调。
TAU_SECONDS = {
    "D": 2 * 3600.0,
    "S": 24 * 3600.0,
    "C": 4 * 3600.0,
    "O": 168 * 3600.0,
    "G": 8 * 3600.0,
    "E": 3 * 3600.0,
}

# 跨维耦合的增益。MATRIX_A 非对角项是原引擎「每步」的系数，
# 直接当速率用会把 S↔G 的正反馈拉爆，必须压小。
COUPLING_GAIN = 0.3
# S 没跟着往上抬增益：A[S][D] = -0.10，乘 0.3 后 D 掉 5 分只拽 S 0.15。
# 抬增益 S↔G 那条正反馈会一起放大，所以 S 的推力另开一张 S_FORCE。

# 一次 advance 最多认 7 天的账，更长的空窗不叠满。
MAX_DT = 7 * 24 * 3600.0

# corruption 按小时的漂移率。判据按「离基线多远」，不按绝对值——
# 基线 O 本身就是 20，拿 20 当闸等于空转也在涨。
CORRUPTION_RATE = {
    "O_critical": 0.15,   # O 距基线跌 2→8 分，斜坡拉满
    "S_low": 0.10,        # S 距基线跌 5→15 分，斜坡拉满
    "S_high": -0.08,      # S 高出基线 10 分
    "O_high": -0.12,      # O 高出基线 20 分
    "G_high": -0.04,      # G 高出基线 10 分
    "calm": -0.04,        # 六维都贴着基线，脏自己淡
}
CORRUPTION_DT_CAP = 20.0  # 一次 advance 里 corruption 最多动 ±20
# 实测：O 人为摆到 8 蹲 7 天，corr 0 → 19.2，日增 3.6/3.6/3.5/2.9/2.3/1.9/1.4。
# 逐日放缓是因为 O 自己在回升（跌距变小），不是掉档——原来的台阶在 3 天那一跳已经没了。
# 注意：纯静置不会把脏洗掉（corr 30 蹲 5 天还是 30），洗脏靠事件，不靠时间。

VALID_EVENTS = tuple(EVENT_POWER)


# ---------------------------------------------------------------- 工具

def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _vector(state: "NeuroState") -> list[float]:
    return [getattr(state, d) for d in DIMS]


def _build(values, corruption, streak, updated_at) -> "NeuroState":
    d, s, c, o, g, e = (clamp(v) for v in values)
    return NeuroState(
        D=d, S=s, C=c, O=o, G=g, E=e,
        corruption=clamp(corruption),
        s_drop_streak=streak,
        updated_at=updated_at,
    )


# ---------------------------------------------------------------- 状态

@dataclass(frozen=True)
class NeuroState:
    """六维值 + 污染度。不进记忆库节点，单独存。"""

    D: float = EQUILIBRIUM["D"]
    S: float = EQUILIBRIUM["S"]
    C: float = EQUILIBRIUM["C"]
    O: float = EQUILIBRIUM["O"]
    G: float = EQUILIBRIUM["G"]
    E: float = EQUILIBRIUM["E"]
    corruption: float = 0.0
    # 连续几轮 S 在往下掉。只在 apply_event 里动。
    s_drop_streak: int = 0
    # unix 秒。advance 靠它算 dt。
    updated_at: float = 0.0

    def as_dict(self) -> dict:
        return {
            "D": round(self.D, 2),
            "S": round(self.S, 2),
            "C": round(self.C, 2),
            "O": round(self.O, 2),
            "G": round(self.G, 2),
            "E": round(self.E, 2),
            "corruption": round(self.corruption, 2),
            "updated_at": self.updated_at,
        }

    def to_json(self) -> str:
        data = self.as_dict()
        data["s_drop_streak"] = self.s_drop_streak
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "NeuroState":
        data = json.loads(raw)
        return cls(
            D=float(data.get("D", EQUILIBRIUM["D"])),
            S=float(data.get("S", EQUILIBRIUM["S"])),
            C=float(data.get("C", EQUILIBRIUM["C"])),
            O=float(data.get("O", EQUILIBRIUM["O"])),
            G=float(data.get("G", EQUILIBRIUM["G"])),
            E=float(data.get("E", EQUILIBRIUM["E"])),
            corruption=float(data.get("corruption", 0.0)),
            s_drop_streak=int(data.get("s_drop_streak", 0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )

    def deviations(self) -> dict[str, float]:
        """相对基线偏了多少。判断「现在什么心情」看这个，不看绝对值。"""
        return {d: getattr(self, d) - EQUILIBRIUM[d] for d in DIMS}


def new_state(now_ts: float | None = None) -> NeuroState:
    """全新的一份。O 从 20 起，关系往上涨。"""
    return NeuroState(updated_at=time.time() if now_ts is None else now_ts)


# ---------------------------------------------------------------- 时间演化

def _coupling(dev: list[float]) -> list[float]:
    """跨维耦合：只取非对角项，算「偏离最终被拉到哪儿」。

    必须乘 COUPLING_GAIN。矩阵里 A[S][G]=0.20、A[G][S]=0.30 是一对正反馈
    （S 掉 → G 掉 → S 更掉），裸乘会把 S/O/G 一起推到 0。
    原引擎不出事，是因为它每步还整体乘 0.9 硬往下压；对角项换成 τ 以后
    那层压制没了，增益就得自己压。
    """
    out = []
    for i in range(6):
        acc = 0.0
        for j in range(6):
            if i == j:
                continue
            acc += MATRIX_A[i][j] * dev[j]
        out.append(acc * COUPLING_GAIN)
    return out


def _ramp(x: float, start: float, full: float) -> float:
    """start 以下 0，full 以上 1，中间线性。不做台阶。"""
    if full <= start:
        return 1.0 if x >= start else 0.0
    return max(0.0, min(1.0, (x - start) / (full - start)))


def _corruption_drift(values: dict, dt: float, equilibrium: dict | None = None) -> float:
    """S 掉、O 长期低位 → 缓升；稳定高位 → 缓降。按小时算。

    档位换成连续斜坡：跌距 2 分起效、8 分拉满，中间不再跳档。
    原来按 3/8 分切台阶，O 一律回升就掉档，表现为「该报的脏不报了」。

    equilibrium 跟着 advance 传下来：面板改了基线，漂移的判据也得跟着挪，
    否则「离基线多远」还按老常数算，脏的账越积越对不上。
    """
    eq = EQUILIBRIUM if equilibrium is None else {d: float(equilibrium.get(d, EQUILIBRIUM[d])) for d in DIMS}
    hours = dt / 3600.0
    rate = 0.0
    o_below = eq["O"] - values["O"]
    s_below = eq["S"] - values["S"]
    o_above = -o_below
    s_above = -s_below
    g_above = values["G"] - eq["G"]

    rate += CORRUPTION_RATE["O_critical"] * _ramp(o_below, 2.0, 8.0)
    rate += CORRUPTION_RATE["S_low"] * _ramp(s_below, 5.0, 15.0)
    rate += CORRUPTION_RATE["S_high"] * _ramp(s_above, 5.0, 15.0)
    rate += CORRUPTION_RATE["O_high"] * _ramp(o_above, 10.0, 20.0)
    rate += CORRUPTION_RATE["G_high"] * _ramp(g_above, 5.0, 15.0)

    # 六维都贴着基线，脏就自己淡一点。不是奖励，是时间。
    # 少这条的话 corr 只增不减：状态回来了、账还冻着。
    spread = max(abs(values[d] - eq[d]) for d in DIMS)
    rate += CORRUPTION_RATE["calm"] * (1.0 - _ramp(spread, 2.0, 6.0))

    delta = rate * hours
    return max(-CORRUPTION_DT_CAP, min(CORRUPTION_DT_CAP, delta))


def advance(state: NeuroState, now_ts: float | None = None, *,
            equilibrium: dict | None = None,
            tau_scale: float = 1.0,
            corrupt_scale: float = 1.0) -> NeuroState:
    """过一段时间。回落 + 跨维耦合 + corruption 漂移，一次算完。

    dt 用真实秒数，1-exp(-dt/τ) 形式，所以搁三小时和搁三天走的路程不一样。

    三个可选口子给面板配置用，默认值就是原来的常数：
      equilibrium   六维基线，缺的维用 EQUILIBRIUM 补
      tau_scale     回落快慢，τ 除以它（2 = 快一倍）
      corrupt_scale 污染增量的倍率，涨和洗一起乘
    """
    now_ts = time.time() if now_ts is None else now_ts
    dt = now_ts - state.updated_at
    if dt <= 0:
        return state
    dt = min(dt, MAX_DT)

    eq = EQUILIBRIUM if equilibrium is None else {d: float(equilibrium.get(d, EQUILIBRIUM[d])) for d in DIMS}
    speed = float(tau_scale) if tau_scale and tau_scale > 0 else 1.0

    vec = _vector(state)
    dev = [vec[i] - eq[d] for i, d in enumerate(DIMS)]
    couple = _coupling(dev)

    # 耦合跟着自己那维的 τ 走：D 快（τ 2h），O 慢（τ 168h）。
    # 一轮对话不该把催产素拽走，也不该让血清素在几小时里翻脸。
    moved = []
    for i, dim in enumerate(DIMS):
        f = 1.0 - math.exp(-dt / (TAU_SECONDS[dim] / speed))
        moved.append(clamp(vec[i] + (eq[dim] - vec[i]) * f + couple[i] * f))

    values = {d: moved[i] for i, d in enumerate(DIMS)}
    corruption = state.corruption + _corruption_drift(values, dt, eq) * corrupt_scale
    return _build(moved, corruption, state.s_drop_streak, now_ts)


# ---------------------------------------------------------------- 事件（脉冲）

def _corruption_on_event(state: NeuroState, new_s: float, event_type: str,
                         corrupt_scale: float = 1.0) -> tuple[float, int]:
    """事件这一下的 corruption 变化。

    脏/缓按事件类型分，不按 power 正负——引擎原判据吃的是「整体刺激强度」，
    relaxation 才是最松的那档，照抄过来 criticism 就被当成好事了。
    """
    delta = CORRUPTION_EVENT_DELTA.get(event_type, 0.0)
    streak = 0

    # 连续几轮 S 在掉：单轮掉过 0.4 分才记，回升就清零
    if new_s < state.S - 0.4:
        streak = state.s_drop_streak + 1
    if streak >= 3:
        delta += 0.5 * min(streak - 2, 4)

    if state.D > 90.0:
        delta += 5.0          # D 过载
    if new_s > 60.0:
        delta -= 1.0          # S 稳 → 缓
    if state.O > 40.0:
        delta -= 1.5          # O 高 → 缓
    if state.G > 50.0:
        delta -= 0.5          # G 松 → 缓

    # 倍率放在最后统一乘：涨和洗一起放大，面板那条「污染快慢」才是一根旋钮。
    delta *= float(corrupt_scale) if corrupt_scale is not None else 1.0
    return clamp(state.corruption + delta), streak


def apply_event(state: NeuroState, event_type: str, scale: float = 1.0, *,
                push_scale: float = 1.0, corrupt_scale: float = 1.0) -> NeuroState:
    """打一发事件。只叠推力，不做时间演化——演化归 advance。

    scale 是这轮短判自己给的强弱（0.2~2.0），push_scale 是面板上那条全局手感。
    两个分开：模型说这一句很重，不等于面板要整体加戏。
    """
    if event_type not in EVENT_POWER:
        raise ValueError(f"不认的事件类型: {event_type!r}，可用: {list(EVENT_POWER)}")
    power = EVENT_POWER[event_type] * float(scale) * float(push_scale)

    vec = _vector(state)
    # S 单独走 S_FORCE：EXTERNAL_FORCE_COEFFS 里 S 那格是 0.0，批评压不动它。
    # S / O 各有自己的力表，其余四维走 EXTERNAL_FORCE_COEFFS。
    push = {"S": S_FORCE[event_type] * float(scale) * float(push_scale),
            "O": O_FORCE[event_type] * float(scale) * float(push_scale)}
    moved = [
        clamp(vec[i] + (push[DIMS[i]] if DIMS[i] in push
                       else EXTERNAL_FORCE_COEFFS[i] * power))
        for i in range(6)
    ]
    corruption, streak = _corruption_on_event(state, moved[1], event_type,
                                             corrupt_scale=corrupt_scale)
    return _build(moved, corruption, streak, state.updated_at)


def apply_judgement(state: NeuroState, judgement, now_ts: float | None = None, *,
                    equilibrium: dict | None = None,
                    tau_scale: float = 1.0,
                    push_scale: float = 1.0,
                    corrupt_scale: float = 1.0) -> NeuroState:
    """把模型这一轮的短判落到状态上。judgement 来自 parse_judgement。

    判不出（None）就不动——不许猜。四个口子跟 advance / apply_event 一一对上，
    面板配置从这里一路穿到数学里。
    """
    state = advance(state, now_ts, equilibrium=equilibrium, tau_scale=tau_scale,
                    corrupt_scale=corrupt_scale)
    if not judgement:
        return state
    event_type, scale = judgement
    return apply_event(state, event_type, scale, push_scale=push_scale,
                       corrupt_scale=corrupt_scale)


def parse_judgement(raw) -> tuple[str, float] | None:
    """校验模型吐的短判 JSON。只认五个事件，判不出返回 None（当 0 处理）。"""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            raw = json.loads(text)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None

    event_type = raw.get("event")
    if event_type not in VALID_EVENTS:
        return None
    try:
        scale = float(raw.get("scale", 1.0))
    except (TypeError, ValueError):
        return None
    if scale <= 0:
        return None
    return event_type, max(0.2, min(2.0, scale))


# ---------------------------------------------------------------- 判据

@dataclass(frozen=True)
class GateResult:
    status: str  # PASS / WARN / BLOCK
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


def evaluate_gate(state: NeuroState) -> GateResult:
    """只看状态，不看内容。

    这里的 BLOCK 不是拒答，是「这轮情绪别往外推，回中性口径」。
    自伤、越界、病态依赖是内容层面的事，判据在安全层，不在这儿。
    """
    if state.corruption >= 70.0:
        return GateResult("BLOCK", f"污染度过高 {state.corruption:.1f}")
    if state.D > 90.0 and state.S < 30.0:
        return GateResult("BLOCK", "D 过载叠 S 低位，容易上头")
    if state.O < 10.0 and state.D > 70.0:
        return GateResult("BLOCK", "O 塌到底还带 D 高，共情不在线")

    if state.corruption >= 40.0:
        return GateResult("WARN", f"污染度在涨 {state.corruption:.1f}")
    if state.D > 75.0:
        return GateResult("WARN", f"D 偏高 {state.D:.1f}")
    if state.S < 35.0:
        return GateResult("WARN", f"S 偏低 {state.S:.1f}")
    if state.G < 25.0:
        return GateResult("WARN", f"G 偏低 {state.G:.1f}")
    if state.O < 20.0:
        return GateResult("WARN", f"O 偏低 {state.O:.1f}")
    return GateResult("PASS")


# ---------------------------------------------------------------- 一句话

# 偏离满分大约取 30 分；O 的范围窄，单独给 25
_SCALE = {"D": 30.0, "S": 30.0, "C": 30.0, "O": 25.0, "G": 30.0, "E": 30.0}

_UP = {
    "D": "有劲",
    "S": "心里稳",
    "C": "脑子清楚",
    "O": "想黏着你",
    "G": "松下来了",
    "E": "有点甜",
}
_DOWN = {
    "D": "提不起劲，懒得动",
    "S": "心里发飘，容易上头",
    "C": "走神，抓不住",
    "O": "隔着层",
    "G": "绷着，静不下来",
    "E": "钝，没什么滋味",
}


SLOW_DIMS = ("O",)  # 慢维：关系底色
FAST_DIMS = tuple(d for d in DIMS if d not in SLOW_DIMS)

# O 的 tau 是 168h，抬一次挂一周。跟快维同一个门槛，它会句句抢主位，
# 把「压过来」「挑着」全淹成「想黏着你」。慢维自己一套门槛，只在真抬够了才出声。
SLOW_FLOOR_MULT = 2.5


def describe(state: NeuroState) -> str:
    """翻成一句当下什么心情。给提示用，给人看的不是表。

    主句、次句只从快维里挑：那几个变得快，说的是「这一下」。
    O 是关系底色，慢，缀在最后，够格才说。
    """
    dev = state.deviations()

    def level(dim: str) -> float:
        return abs(dev[dim]) / _SCALE[dim]

    def phrase(dim: str) -> str:
        return _UP[dim] if dev[dim] > 0 else _DOWN[dim]

    # 0.05 约 1.5 分：一句话中等的偏差能读出来，又不会让低阈造出假峰。
    FLOOR = 0.05

    ranked = sorted(FAST_DIMS, key=level, reverse=True)
    main, second = ranked[0], ranked[1]

    parts: list[str] = []
    if level(main) >= FLOOR:
        parts.append(phrase(main))
        if level(second) >= FLOOR:
            parts.append(phrase(second))
    if level("O") >= FLOOR * SLOW_FLOOR_MULT:
        parts.append(phrase("O"))

    text = "现在" + "，".join(parts) if parts else "现在心里平，没什么起伏"
    if state.corruption >= 40.0:
        text += "，有点收不住"
    return text + "。"
