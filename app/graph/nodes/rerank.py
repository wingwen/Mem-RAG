"""Rerank：对宽召回结果做 Cross-Encoder 风格精排。"""

from app.core import config_data as config
from app.graph.state import AgentState
from app.retrieval.rerank import RerankService


def rerank_node(state: AgentState) -> dict:
    """A3 精排节点：宽召回候选 → Rerank Top-N（生产必经）。"""
    query = state.get("rewritten_query") or state.get("query") or ""
    docs = state.get("retrieved_docs") or []

    if not config.RERANK_ENABLED:
        return {
            "retrieved_docs": docs[: config.SIMILARITY_THRESHOLD],
            "retrieval_status": ["[状态] Rerank 未启用，应由 retrieval 走 A1\n"],
        }

    if not docs:
        return {
            "retrieved_docs": [],
            "retrieval_status": ["[状态] 无候选文档，跳过 Rerank\n"],
        }

    reranked = RerankService().rerank(query, docs, top_n=config.RERANK_TOP_N)
    return {
        "retrieved_docs": reranked,
        "retrieval_status": [
            f"[状态] A3 Rerank 完成 ({config.RERANK_MODEL})，"
            f"{len(docs)} 候选 → Top-{len(reranked)}\n"
        ],
    }
