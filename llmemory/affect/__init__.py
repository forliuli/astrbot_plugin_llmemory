"""情绪与内分泌层。

对外只暴露状态模型和一句话注入，其余别从这里拿。
"""

from .config import AffectConfig, load_config
from .judge_model import chat, judge_model
from .judge_rule import Verdict, judge, judge_tuple
from .neurostate import (
    DIMS,
    DIM_NAMES,
    EQUILIBRIUM,
    NeuroState,
    advance,
    apply_event,
    apply_judgement,
    describe,
    evaluate_gate,
    new_state,
    parse_judgement,
)

__all__ = [
    "DIMS",
    "DIM_NAMES",
    "EQUILIBRIUM",
    "AffectConfig",
    "NeuroState",
    "Verdict",
    "advance",
    "apply_event",
    "apply_judgement",
    "describe",
    "evaluate_gate",
    "judge",
    "chat",
    "judge_model",
    "judge_tuple",
    "load_config",
    "new_state",
    "parse_judgement",
]
