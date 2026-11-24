"""配置容错层。

面板给的是人填的字典：漏项、填错类型、写个 999 都可能。这一层只管一件事——
不管喂进来什么，出去的都是能直接塞进数学里的值。坏的一律退回默认，能夹的夹紧。

不在这儿做判断，也不报错。插件启动时不该因为用户把「回落快慢」填成 "快一点" 就崩。
"""

from __future__ import annotations

import json
import os

from ..paths import under
from .neurostate import DIMS, EQUILIBRIUM

# 面板字段 → 六维。名字是人看的，DIMS 是数学看的。
BASE_FIELDS = {
    "adv_base_drive": "D",
    "adv_base_stable": "S",
    "adv_base_focus": "C",
    "adv_base_close": "O",
    "adv_base_relax": "G",
    "adv_base_comfort": "E",
}

DEFAULTS = {
    "enable": True,
    "judge_mode": "rule",
    "model_url": "",
    "model_key": "",
    "model_name": "",
    "affect_tone": True,
    "affect_recall": False,
    "adv_decay_speed": 1.0,
    "adv_push_scale": 1.0,
    "adv_corrupt_speed": 1.0,
    "adv_state_file": "",
}

# 夹紧区间。下限不是 0：倍率给 0 等于把这条通路关死，用户没那么说过。
LIMITS = {
    "adv_decay_speed": (0.1, 10.0),
    "adv_push_scale": (0.1, 5.0),
    "adv_corrupt_speed": (0.0, 5.0),
}
BASE_RANGE = (0.0, 100.0)

JUDGE_MODES = ("rule", "model", "off")


def _as_bool(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "1", "yes", "on", "开"):
            return True
        if low in ("false", "0", "no", "off", "关", ""):
            return False
    return default


def _as_float(v, default, lo=None, hi=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):   # NaN / inf
        return default
    if lo is not None:
        f = max(lo, f)
    if hi is not None:
        f = min(hi, f)
    return f


def _as_str(v, default=""):
    if isinstance(v, str):
        return v.strip()
    if v is None:
        return default
    return str(v).strip()


class AffectConfig:
    """一份能直接用的配置。字段名跟面板一致，省得两头对。"""

    __slots__ = ("enable", "judge_mode", "model_url", "model_key", "model_name",
                 "affect_tone", "affect_recall", "equilibrium",
                 "tau_scale", "push_scale", "corrupt_scale", "state_file",
                 "_raw_state_file")

    def __init__(self, raw=None, base_dir: str = ""):
        raw = raw if isinstance(raw, dict) else {}

        self.enable = _as_bool(raw.get("enable"), DEFAULTS["enable"])
        self.affect_tone = _as_bool(raw.get("affect_tone"), DEFAULTS["affect_tone"])
        self.affect_recall = _as_bool(raw.get("affect_recall"), DEFAULTS["affect_recall"])

        mode = _as_str(raw.get("judge_mode"), DEFAULTS["judge_mode"]).lower()
        self.judge_mode = mode if mode in JUDGE_MODES else DEFAULTS["judge_mode"]

        self.model_url = _as_str(raw.get("model_url"))
        self.model_key = _as_str(raw.get("model_key"))
        self.model_name = _as_str(raw.get("model_name"))
        # 选了模型却没把三样填全：悄悄退回规则。不报错，也不留半截状态。
        if self.judge_mode == "model" and not (self.model_url and self.model_key and self.model_name):
            self.judge_mode = "rule"

        self.equilibrium = {}
        for field, dim in BASE_FIELDS.items():
            self.equilibrium[dim] = _as_float(raw.get(field), EQUILIBRIUM[dim], *BASE_RANGE)
        for dim in DIMS:                       # 表里没有的维也不许缺
            self.equilibrium.setdefault(dim, EQUILIBRIUM[dim])

        for name in ("adv_decay_speed", "adv_push_scale", "adv_corrupt_speed"):
            lo, hi = LIMITS[name]
            setattr(self, {"adv_decay_speed": "tau_scale",
                           "adv_push_scale": "push_scale",
                           "adv_corrupt_speed": "corrupt_scale"}[name],
                    _as_float(raw.get(name), DEFAULTS[name], lo, hi))

        self._raw_state_file = _as_str(raw.get("adv_state_file"))
        self.state_file = self._resolve_path(self._raw_state_file, base_dir)

    @staticmethod
    def _resolve_path(path: str, base_dir: str) -> str:
        if not path:
            path = under("neurostate.json")
        if not os.path.isabs(path) and base_dir:
            path = os.path.join(base_dir, path)
        return os.path.normpath(path)

    # ---- 给数学层的口子 ------------------------------------------------

    def advance_kwargs(self) -> dict:
        """advance 要的三个参数。"""
        return {"equilibrium": dict(self.equilibrium),
                "tau_scale": self.tau_scale,
                "corrupt_scale": self.corrupt_scale}

    def event_kwargs(self) -> dict:
        """apply_event 要的。"""
        return {"push_scale": self.push_scale, "corrupt_scale": self.corrupt_scale}

    def judge_kwargs(self) -> dict:
        """apply_judgement 一次给全。"""
        kw = self.advance_kwargs()
        kw["push_scale"] = self.push_scale
        return kw

    # ---- 存盘 ----------------------------------------------------------

    def load_state(self):
        """读存盘。文件不在 / 坏了 / 字段不认，都从基线重新起，不抛。"""
        from .neurostate import NeuroState
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return NeuroState()
            fields = {}
            for dim in DIMS:
                if dim in data:
                    fields[dim] = _as_float(data[dim], self.equilibrium[dim], 0.0, 100.0)
            if "corruption" in data:
                fields["corruption"] = _as_float(data["corruption"], 0.0, 0.0, 100.0)
            if "s_drop_streak" in data:
                try:
                    fields["s_drop_streak"] = max(0, int(data["s_drop_streak"]))
                except (TypeError, ValueError):
                    pass
            if "updated_at" in data:
                fields["updated_at"] = _as_float(data["updated_at"], 0.0)
            return NeuroState(**fields)
        except (OSError, ValueError, TypeError):
            return NeuroState()

    def save_state(self, state) -> bool:
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            payload = {dim: float(getattr(state, dim)) for dim in DIMS}
            payload["corruption"] = float(state.corruption)
            payload["s_drop_streak"] = int(state.s_drop_streak)
            payload["updated_at"] = float(state.updated_at)
            tmp = self.state_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.state_file)
            return True
        except (OSError, TypeError, ValueError):
            return False


def load_config(raw=None, base_dir: str = "") -> AffectConfig:
    return AffectConfig(raw, base_dir)
