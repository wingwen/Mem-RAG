"""Retrieval：Hybrid Dense+Sparse → Merge → Rerank。"""

from app.graph.state import AgentState
from app.retrieval.gate import check_retrieval_hit
from app.retrieval.service import RetrievalService

# 进程级单例，与 RagService 共享时可注入
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
    statuses, docs = svc.hybrid_search(query)
    hit = check_retrieval_hit(query, docs)
    extra = ""
    if not hit:
        extra = "[状态] 检索结果与问题关键词不匹配，视为未命中\n"
    return {
        "retrieved_docs": docs if hit else [],
        "retrieval_hit": hit,
        "retrieval_status": statuses + ([extra] if extra else []),
    }
