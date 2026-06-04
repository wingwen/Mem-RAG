"""Generation：基于 final_context 生成回答。"""

from __future__ import annotations

from typing import AsyncIterator

from app.core.prompts import rag_prompt_template
from app.graph.state import AgentState
from app.llm.streaming import astream_chat


def _build_messages(state: AgentState):
    query = state.get("query") or ""
    memory_block = state.get("memory_block") or ""
    final_context = state.get("final_context") or "无相关参考资料"
    history = state.get("history_messages") or []

    ref_only = final_context
    if "【检索参考资料】" in final_context:
        ref_only = final_context.split("【检索参考资料】", 1)[-1].strip()

    return rag_prompt_template.format_messages(
        memory_context=memory_block,
        context=ref_only,
        input=query,
        history=history,
    )


def generation_node(state: AgentState) -> dict:
    """同步全量生成（用于 graph.invoke）。"""
    from app.core import config_data as config
    from app.llm.streaming import _lc_messages_to_dashscope, _sync_invoke_generate

    messages = _build_messages(state)
    ds_messages = _lc_messages_to_dashscope(messages)
    answer = _sync_invoke_generate(ds_messages, config.CHAT_MODEL)
    return {"answer": answer or ""}


async def stream_generation(state: AgentState) -> AsyncIterator[str]:
    """流式生成（DashScope 原生 API，避免 LangChain astream KeyError）。"""
    messages = _build_messages(state)
    async for text in astream_chat(messages):
        if text:
            yield text
