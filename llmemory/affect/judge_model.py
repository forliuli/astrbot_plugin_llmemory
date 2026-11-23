"""模型短判 + 通用直连。

只走 urllib，不引第三方库。任何异常都返回 None / 空串，让调用方退回查词表——
判不出比判错好，比把异常抛进对话里好。

spark-x2.6-4b 这类是思考模型：不关思考，64 token 全花在 reasoning 上，正文空。
所以默认带 thinking: disabled，遇到不认这字段的服务（400/422）再退一次不带。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .neurostate import parse_judgement

TIMEOUT = 15.0
MAX_TOKENS = 256
MAX_CHARS = 500

_SYSTEM = "\n".join([
    "你在读一个人对助手说的话，判它属于哪一类。只输出一行 JSON，不要解释。",
    "event 只能取这五个之一：praise 夸 / criticism 批 / bonding 亲近 / stress 压 / relaxation 松。",
    "scale 是强度，0.2 到 2.0，1.0 是平常。",
    '比如 {"event": "praise", "scale": 1.0}',
    '判不出就输出 {"event": "none"}。',
])


def _post(url: str, payload: dict, key: str, timeout: float) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def chat(url, key, name, messages, max_tokens=MAX_TOKENS, timeout=TIMEOUT,
         temperature=0.7):
    """直连一条 chat/completions，回正文。先关思考，服务不认就不带那字段重发一次。"""
    payload = {
        "model": name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "thinking": {"type": "disabled"},
    }
    try:
        raw = _post(url, payload, key, timeout)
    except urllib.error.HTTPError as e:
        if e.code not in (400, 404, 422):
            raise
        payload.pop("thinking", None)
        raw = _post(url, payload, key, timeout)
    obj = json.loads(raw.decode("utf-8"))
    choices = obj.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    msg = choices[0].get("message")
    if isinstance(msg, dict):
        return (msg.get("content") or "").strip()
    return (choices[0].get("text") or "").strip()


def _extract(raw: bytes):
    """从回包里抠出 judgement：OpenAI 风格的走 choices，直给的直接认。"""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return obj if "event" in obj else None
    first = choices[0] if isinstance(choices[0], dict) else None
    if not first:
        return None
    msg = first.get("message")
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"]
    if "text" in first:
        return first["text"]
    return None


def judge_model(text: str, cfg, timeout: float = TIMEOUT):
    """text → (event, scale) 或 None。cfg 要带 model_url / model_key / model_name。"""
    if not text:
        return None
    url = getattr(cfg, "model_url", "") or ""
    key = getattr(cfg, "model_key", "") or ""
    name = getattr(cfg, "model_name", "") or ""
    if not (url and name):
        return None

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": text[:MAX_CHARS]},
    ]
    try:
        got = chat(url, key, name, messages, MAX_TOKENS, timeout, 0.0)
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None
    return parse_judgement(got)
