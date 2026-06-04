"""LangGraph Agentic RAG 状态机编译。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.core import config_data as config
from app.graph.nodes import (
    fusion_node,
    generation_node,
    memory_node,
    rerank_node,
    retrieval_node,
    router_node,
)
from app.graph.state import AgentState

_compiled_graph = None


def build_agent_graph(*, include_generation: bool = True, include_rerank: bool | None = None):
    """
    START → Router → Memory → Retrieval → [Rerank] → Fusion → [Generation] → END
    """
    use_rerank = config.RERANK_ENABLED if include_rerank is None else include_rerank
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("memory", memory_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("fusion", fusion_node)

    builder.add_edge(START, "router")
    builder.add_edge("router", "memory")
    builder.add_edge("memory", "retrieval")

    if use_rerank:
        builder.add_node("rerank", rerank_node)
        builder.add_edge("retrieval", "rerank")
        builder.add_edge("rerank", "fusion")
    else:
        builder.add_edge("retrieval", "fusion")

    if include_generation:
        builder.add_node("generation", generation_node)
        builder.add_edge("fusion", "generation")
        builder.add_edge("generation", END)
    else:
        builder.add_edge("fusion", END)

    return builder.compile()


def get_agent_graph(*, include_generation: bool = False):
    """API 流式默认编译到 Fusion；批量推理可含 Generation。"""
    global _compiled_graph
    if include_generation:
        return build_agent_graph(include_generation=True)
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph(include_generation=False)
    return _compiled_graph
