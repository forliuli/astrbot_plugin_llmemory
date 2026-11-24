"""数据目录只在这一处定。

规范位置是 data/plugin_data/<插件名>，由框架的 StarTools.get_data_dir 给。
配了 LLMEMORY_DATA_DIR 就听它的，那是显式挪库的口子，不是默认。
别处不再拼 "plugin_data/..." 这种相对路径。
"""

from __future__ import annotations

import os

PLUGIN_NAME = "astrbot_plugin_llmemory"


def data_dir() -> str:
    """插件的数据目录，绝对路径，框架会顺手建好。"""
    env = os.environ.get("LLMEMORY_DATA_DIR", "").strip()
    if env:
        return os.path.abspath(env)
    from astrbot.api.star import StarTools

    return str(StarTools.get_data_dir(PLUGIN_NAME))


def under(*parts: str) -> str:
    """数据目录下面某个路径。父目录不建，谁写谁建。"""
    return os.path.join(data_dir(), *parts)
