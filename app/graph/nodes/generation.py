"""Generation：基于 final_context 生成回答。"""

from __future__ import annotations

from typing import AsyncIterator

from app.core.prompts import rag_prompt_template
from app.graph.state import AgentState
from app.llm.factory import get_chat_model


def _extract_chunk_content(chunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content) if content else ""


def generation_node(state: AgentState) -> dict:
    """同步全量生成（用于 graph.invoke）。"""
    from app.core import config_data as config

    if not state.get("retrieval_hit", True):
        return {"answer": state.get("answer") or config.NO_KB_ANSWER_MESSAGE}

    query = state.get("query") or ""
    memory_block = state.get("memory_block") or ""
    final_context = state.get("final_context") or "无相关参考资料"
    history = state.get("history_messages") or []

    # final_context 已含记忆+资料；system 仍单独注入 memory_block 保持模板兼容
    ref_only = final_context
    if "【检索参考资料】" in final_context:
        ref_only = final_context.split("【检索参考资料】", 1)[-1].strip()

    messages = rag_prompt_template.format_messages(
        memory_context=memory_block,
        context=ref_only,
        input=query,
        history=history,
    )
    model = get_chat_model(streaming=False)
    resp = model.invoke(messages)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return {"answer": answer or ""}


async def stream_generation(state: AgentState) -> AsyncIterator[str]:
    """流式生成（API 层在 Fusion 之后调用）。"""
    from app.core import config_data as config

    if not state.get("retrieval_hit", True):
        yield state.get("answer") or config.NO_KB_ANSWER_MESSAGE
        return

    query = state.get("query") or ""
    memory_block = state.get("memory_block") or ""
    final_context = state.get("final_context") or "无相关参考资料"
    history = state.get("history_messages") or []

    ref_only = final_context
    if "【检索参考资料】" in final_context:
        ref_only = final_context.split("【检索参考资料】", 1)[-1].strip()

    messages = rag_prompt_template.format_messages(
        memory_context=memory_block,
        context=ref_only,
        input=query,
        history=history,
    )
    model = get_chat_model(streaming=True)
    async for chunk in model.astream(messages):
        text = _extract_chunk_content(chunk)
        if text:
            yield text
