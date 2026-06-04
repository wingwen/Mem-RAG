"""LangGraph Agentic RAG 状态机编译。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    fusion_node,
    generation_node,
    memory_node,
    no_answer_node,
    retrieval_node,
    rewrite_node,
)
from app.graph.state import AgentState

_compiled_graph = None


def _route_after_retrieval(state: AgentState) -> str:
    if state.get("retrieval_hit", False):
        return "fusion"
    return "no_answer"


def build_agent_graph(*, include_generation: bool = True):
    """
    START → Rewrite → Memory → Retrieval → Fusion|NoAnswer → [Generation] → END
    """
    builder = StateGraph(AgentState)

    builder.add_node("rewrite", rewrite_node)
    builder.add_node("memory", memory_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("fusion", fusion_node)
    builder.add_node("no_answer", no_answer_node)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "memory")
    builder.add_edge("memory", "retrieval")
    builder.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {"fusion": "fusion", "no_answer": "no_answer"},
    )
    builder.add_edge("no_answer", END)

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
