"""数据目录只在这一处定。

规范位置是 data/plugin_data/<插件名>，由框架的 StarTools.get_data_dir 给。
配了 LLMEMORY_DATA_DIR 就听它的——那是显式挪库的口子，不是默认。
别处不再拼 "plugin_data/..." 这种相对路径：那种写法跟着进程工作目录走，
会落到 AstrBot 根下面，跟插件数据目录不是一回事。
"""

from __future__ import annotations

import os

PLUGIN_NAME = "astrbot_plugin_llmemory"


def data_dir() -> str:
    """插件的数据目录，绝对路径，框架会顺手建好。"""
    env = os.environ.get("LLMEMORY_DATA_DIR", "").strip()
    if env:
        return os.path.abspath(env)
    try:
        from astrbot.api.star import StarTools

        return str(StarTools.get_data_dir(PLUGIN_NAME))
    except Exception:
        # 框架老得没有 StarTools：按规范位置拼一份，也不回插件目录里去
        return os.path.abspath(os.path.join("data", "plugin_data", PLUGIN_NAME))


def under(*parts: str) -> str:
    """数据目录下面某个路径。父目录不建，谁写谁建。"""
    return os.path.join(data_dir(), *parts)
