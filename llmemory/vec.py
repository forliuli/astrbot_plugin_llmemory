# -*- coding: utf-8 -*-
"""llmemory 的向量入口：一段话 -> 512 维。

模型 bge-small-zh-v1.5 的 ONNX 版不跟着包走（九十多兆，塞不进市场 16MB），
第一次要用的时候由 model_fetch 按地址下到 models/。CPU 推理，不联网，不花 token。
取 [CLS] 池化 + L2 归一化，于是余弦相似度就是点积。

依赖缺了不算错：onnxruntime / tokenizers 没装上，这里就安安静静地坐着，
ready 一直是 False，检索退成时间窗，其余照跑。
"""

import importlib.util
import os
import time

import numpy as np

MAX_LEN = 512
BATCH = 16

# onnxruntime 1.19.2 在系统缺 VC++ 运行库时，导入就会崩——先把 capi 目录里那份 DLL 挂上。
_spec = importlib.util.find_spec("onnxruntime")
if _spec and _spec.submodule_search_locations:
    _capi = os.path.join(list(_spec.submodule_search_locations)[0], "capi")
    if os.path.isdir(_capi):
        try:
            os.add_dll_directory(_capi)
        except Exception:
            pass

# 这两个包没装，只该少一层，不该让整个插件加载失败。
_DEPS_ERR = ""
try:
    import onnxruntime as ort                    # noqa: E402
except Exception as e:                           # noqa: BLE001
    ort = None
    _DEPS_ERR = "onnxruntime 没装上（%s: %s）" % (type(e).__name__, e)
try:
    from tokenizers import Tokenizer             # noqa: E402
except Exception as e:                           # noqa: BLE001
    Tokenizer = None
    if not _DEPS_ERR:
        _DEPS_ERR = "tokenizers 没装上（%s: %s）" % (type(e).__name__, e)


def deps_error() -> str:
    """缺什么依赖；齐了就是空串。"""
    return _DEPS_ERR


class Encoder:
    """懒加载；模型不在就安安静静地缺着，检索那头退化成时间窗。"""

    def __init__(self, model_dir, threads=None):
        self.dir = model_dir
        self.onnx = os.path.join(model_dir, "model.onnx")
        self.tok_file = os.path.join(model_dir, "tokenizer.json")
        self.threads = threads
        self.tok = None
        self.sess = None
        self.names = None
        self.dim = 0
        self._qcache = {}

    @property
    def deps_ok(self):
        return ort is not None and Tokenizer is not None

    @property
    def ready(self):
        return (self.deps_ok and os.path.isfile(self.onnx)
                and os.path.isfile(self.tok_file))

    def load(self, threads=None):
        if self.sess is not None:
            return self
        if not self.deps_ok:
            raise RuntimeError(_DEPS_ERR or "向量依赖没装")
        tok = Tokenizer.from_file(self.tok_file)
        tok.enable_truncation(max_length=MAX_LEN)
        tok.enable_padding(pad_id=0, pad_token="[PAD]")

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or self.threads or min(8, os.cpu_count() or 4)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(self.onnx, so, providers=["CPUExecutionProvider"])
        self.tok = tok
        self.sess = sess
        self.names = [i.name for i in sess.get_inputs()]
        self.dim = sess.get_outputs()[0].shape[-1]
        return self

    def encode(self, texts, batch=BATCH, quiet=True):
        """一批文本 -> [N, dim] float32，已 L2 归一化。"""
        if not self.ready:
            return None
        self.load()
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        t0 = time.time()
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            encs = self.tok.encode_batch(part)
            maxlen = max(len(e.ids) for e in encs)
            ids = np.zeros((len(encs), maxlen), dtype=np.int64)
            am = np.zeros((len(encs), maxlen), dtype=np.int64)
            for j, e in enumerate(encs):
                n = len(e.ids)
                ids[j, :n] = e.ids
                am[j, :n] = e.attention_mask
            feed = {"input_ids": ids, "attention_mask": am,
                    "token_type_ids": np.zeros((len(encs), maxlen), dtype=np.int64)}
            feed = {k: v for k, v in feed.items() if k in self.names}
            h = self.sess.run(None, feed)[0]
            v = h[:, 0, :]                      # bge 用 [CLS]
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            out[i:i + len(part)] = v.astype(np.float32)
            if not quiet and (i // batch) % 20 == 19:
                done = i + len(part)
                rate = done / max(time.time() - t0, 1e-6)
                print("  %5d/%d  %.0f 块/秒" % (done, len(texts), rate), flush=True)
        return out

    def one(self, text):
        """一句话 -> 一条归一化向量；同一句回来直接用上次的。"""
        key = text.strip()
        if not key:
            return None
        hit = self._qcache.get(key)
        if hit is not None:
            return hit
        v = self.encode([key], quiet=True)
        if v is None:
            return None
        v = v[0]
        if len(self._qcache) > 512:
            self._qcache.clear()
        self._qcache[key] = v
        return v
