"""DashScope 流式生成（绕过 LangChain astream 与 HTTPError 包装缺陷）。"""

from __future__ import annotations

import asyncio
import threading
from http import HTTPStatus
from typing import AsyncIterator, Iterable

import dashscope
from dashscope import Generation
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core import config_data as config
from app.core.logger import logger

dashscope.api_key = config.DASHSCOPE_API_KEY


def _generation_extra_params(model_name: str) -> dict:
    if "qwen3" in (model_name or "").lower():
        return {"enable_thinking": False}
    return {}


def _lc_messages_to_dashscope(messages: Iterable[BaseMessage]) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "user"
        content = msg.content
        if isinstance(content, list):
            content = "".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        out.append({"role": role, "content": str(content)})
    return out


def _extract_delta(resp) -> str:
    try:
        output = resp.output
        if not output:
            return ""
        choices = output.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content") or ""
    except Exception:
        return ""


def _format_api_error(resp) -> str:
    code = getattr(resp, "code", "") or ""
    message = getattr(resp, "message", "") or ""
    status = getattr(resp, "status_code", "?")
    return (
        f"DashScope 调用失败 (HTTP {status}): {code} — {message}。"
        f"请检查 DASHSCOPE_API_KEY、模型「{config.CHAT_MODEL}」是否开通及账户余额。"
    )


def _sync_stream_generate(messages: list[dict], model: str):
    responses = Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        stream=True,
        incremental_output=True,
        **_generation_extra_params(model),
    )
    for resp in responses:
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(_format_api_error(resp))
        delta = _extract_delta(resp)
        if delta:
            yield delta


def _sync_invoke_generate(messages: list[dict], model: str) -> str:
    resp = Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        **_generation_extra_params(model),
    )
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(_format_api_error(resp))
    return _extract_delta(resp) or ""


async def astream_chat(
    messages: Iterable[BaseMessage],
    *,
    model: str | None = None,
) -> AsyncIterator[str]:
    """DashScope 原生流式；失败则整段 invoke。"""
    model_name = model or config.CHAT_MODEL
    ds_messages = _lc_messages_to_dashscope(messages)

    if not config.DASHSCOPE_API_KEY:
        yield "\n[错误] 未配置 DASHSCOPE_API_KEY，请在 .env 中设置。\n"
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for piece in _sync_stream_generate(ds_messages, model_name):
                asyncio.run_coroutine_threadsafe(queue.put(piece), loop).result(timeout=120)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=10)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result(timeout=10)

    threading.Thread(target=worker, daemon=True).start()

    stream_ok = False
    while True:
        item = await queue.get()
        if item is None:
            stream_ok = True
            break
        if isinstance(item, Exception):
            logger.warning(f"[LLM] 流式失败，回退 invoke: {item}")
            try:
                text = await asyncio.to_thread(
                    _sync_invoke_generate, ds_messages, model_name
                )
                if text:
                    yield text
            except Exception as invoke_exc:
                logger.error(f"[LLM] invoke 失败: {invoke_exc}")
                yield f"\n[错误] {invoke_exc}\n"
            return
        stream_ok = True
        yield item

    if not stream_ok:
        yield "\n[错误] 模型未返回内容。\n"


async def ainvoke_chat(
    messages: Iterable[BaseMessage],
    *,
    model: str | None = None,
) -> str:
    model_name = model or config.CHAT_MODEL
    ds_messages = _lc_messages_to_dashscope(messages)
    return await asyncio.to_thread(_sync_invoke_generate, ds_messages, model_name)
