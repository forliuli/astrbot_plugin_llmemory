# -*- coding: utf-8 -*-
"""模型文件不在包里，第一次要用的时候自己下来。

包里只带 tokenizer.json（四百多 K）。model.onnx 九十多兆，塞不进市场的 16MB，
所以走源：先试第一个地址，不成换下一个。下不下来都不挡事——vec 那头会退成
时间窗检索，插件照样起、照样记，只是取回那层糙一点。

几个约定：
  · 先写 .part，校验过了再 os.replace 上名，中途断了不会留个半截的当好的用。
  · 同一时间只许一个线程下（.lock 文件独占创建），别几个协程一起扑上去。
  · 地址和哈希都从外面传，代码里不写死任何 URL——填错的地址比没有更坏。
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import urllib.request

from astrbot.api import logger

ONNX_NAME = "model.onnx"
TOK_NAME = "tokenizer.json"

UA = "astrbot-plugin-llmemory/0.1 (+https://github.com/)"
MIN_ONNX_BYTES = 8 * 1024 * 1024      # 比这小的，肯定不是好文件
CHUNK = 1 << 20

_running = threading.Lock()
_started = False


def sha256_of(path: str, bufsize: int = CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ready(model_dir: str) -> bool:
    """两样都在才算齐。tokenizer 一般随包带，缺了也补不了，一样算没齐。"""
    return (os.path.isfile(os.path.join(model_dir, ONNX_NAME))
            and os.path.isfile(os.path.join(model_dir, TOK_NAME)))


def _fetch_one(url: str, dst: str, sha: str = "", timeout: int = 15) -> bool:
    """一个地址下一次。成了 True，没成 False，绝不抛到外面。"""
    tmp = dst + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            last = 0.0
            with open(tmp, "wb") as f:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
                    now = time.time()
                    if now - last > 3.0:
                        last = now
                        if total:
                            logger.info(f"[llmemory] 模型下载 {got * 100 // total}% "
                                        f"（{got >> 20}/{total >> 20} MB）")
                        else:
                            logger.info(f"[llmemory] 模型下载进度：{got >> 20} MB")
        size = os.path.getsize(tmp)
        if size < MIN_ONNX_BYTES:
            logger.warning(f"[llmemory] {url} 返回内容过小（{size} 字节），判定无效并丢弃。")
            os.remove(tmp)
            return False
        if total and size != total:
            logger.warning(f"[llmemory] 长度校验不符（{size}/{total}），丢弃。")
            os.remove(tmp)
            return False
        if sha:
            got_sha = sha256_of(tmp)
            if got_sha.lower() != sha.lower():
                logger.warning("[llmemory] sha256 校验不符，丢弃。"
                               f"下到的是 {got_sha[:16]}…，要的是 {sha[:16]}…")
                os.remove(tmp)
                return False
        os.replace(tmp, dst)
        return True
    except Exception as e:
        logger.warning(f"[llmemory] 从 {url} 下载模型失败：{type(e).__name__}: {e}")
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def fetch(model_dir: str, urls, sha: str = "", timeout: int = 15) -> bool:
    """按顺序试地址，成一次就停。全败返回 False，不抛。"""
    dst = os.path.join(model_dir, ONNX_NAME)
    if os.path.isfile(dst):
        return True
    if not urls:
        logger.info("[llmemory] 未配置模型下载地址，向量层停用，检索降级为时间窗。")
        return False
    try:
        os.makedirs(model_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"[llmemory] 创建模型目录失败：{e}")
        return False
    for url in urls:
        url = str(url or "").strip()
        if not url:
            continue
        logger.info(f"[llmemory] 尝试下载：{url}")
        if _fetch_one(url, dst, sha=sha, timeout=timeout):
            logger.info("[llmemory] 模型下载完成，向量层于下次调用时可用。")
            return True
    logger.warning("[llmemory] 所有下载地址均失败，检索降级为时间窗，不影响插件运行。")
    return False


def fetch_in_background(model_dir: str, urls, sha: str = "",
                        timeout: int = 15, on_done=None) -> bool:
    """起个线程下，不挡启动。已经在下了就返回 False，不重复起。"""
    global _started
    if ready(model_dir):
        return False
    if _started or not _running.acquire(blocking=False):
        return False
    lock = os.path.join(model_dir, ".lock")
    try:
        os.makedirs(model_dir, exist_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        _running.release()
        return False
    except Exception:
        _running.release()
        return False

    def _run():
        global _started
        try:
            ok = fetch(model_dir, urls, sha=sha, timeout=timeout)
            if ok and on_done:
                try:
                    on_done()
                except Exception as e:
                    logger.warning(f"[llmemory] 模型就绪回调出错：{e}")
        finally:
            _started = False
            try:
                os.remove(lock)
            except Exception:
                pass
            _running.release()

    _started = True
    t = threading.Thread(target=_run, name="llmemory-model-fetch", daemon=True)
    t.start()
    return True
