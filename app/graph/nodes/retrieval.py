"""Retrieval：混合检索宽召回（候选集，精排由 rerank 节点完成）。"""

from app.core import config_data as config
from app.graph.state import AgentState
from app.retrieval.pipeline import production_pipeline_config, recall_candidates, retrieve_documents
from app.retrieval.service import RetrievalService

_retrieval_svc: RetrievalService | None = None


def get_retrieval_service(vector_service=None) -> RetrievalService:
    global _retrieval_svc
    if vector_service is not None:
        return RetrievalService(vector_service=vector_service)
    if _retrieval_svc is None:
        _retrieval_svc = RetrievalService()
    return _retrieval_svc


def set_retrieval_service(svc: RetrievalService) -> None:
    global _retrieval_svc
    _retrieval_svc = svc


def retrieval_node(state: AgentState) -> dict:
    query = state.get("rewritten_query") or state.get("query") or ""
    svc = get_retrieval_service()
    vs = svc._vector

    if config.RERANK_ENABLED:
        # A3：仅召回候选，禁止在此截断为 Top-K
        statuses = [
            "[状态] 正在执行混合检索 (Milvus + BM25 + RRF)...\n",
            f"[状态] A3 宽召回 Top-{config.RETRIEVAL_RECALL_K}（待 Rerank 精排）...\n",
        ]
        docs = recall_candidates(query, vs)
        statuses.append(f"[状态] 召回 {len(docs)} 条候选，进入 Rerank...\n")
        return {"retrieved_docs": docs, "retrieval_status": statuses}

    # RERANK 关闭：整条链路走 A1 legacy，跳过宽召回无精排
    statuses = [
        "[状态] Rerank 已关闭，使用 hybrid_legacy (A1) 检索...\n",
    ]
    cfg = production_pipeline_config()
    docs = retrieve_documents(query, vs, pipeline=cfg)
    statuses.append(f"[状态] 返回 Top-{len(docs)} 条文档\n")
    return {"retrieved_docs": docs, "retrieval_status": statuses}
