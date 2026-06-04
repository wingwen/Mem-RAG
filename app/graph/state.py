"""LangGraph Agent 状态定义。"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


def _merge_status(left: list[str] | None, right: list[str] | None) -> list[str]:
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    """Agentic RAG 工作流共享状态。"""

    # --- 输入 ---
    query: str
    session_id: str

    # --- Query Rewrite ---
    rewritten_query: str
    route_decision: str  # rewrite_applied | rewrite_skip | rewrite_fallback | empty

    # --- Memory ---
    memories: list[dict[str, Any]]
    memory_block: str
    history_messages: list[BaseMessage]
    active_topic: str

    # --- Retrieval ---
    retrieved_docs: list[Document]
    retrieval_hit: bool
    retrieval_status: Annotated[list[str], _merge_status]

    # --- Fusion ---
    final_context: str

    # --- Generation ---
    answer: str
