"""可配置的检索流水线（供 API 与消融评估共用）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document

from app.core import config_data as config
from app.core.logger import logger
from app.core.vector_stores import VectorStoreService
from app.retrieval.rerank import RerankService

RetrievalMode = Literal["dense", "hybrid", "hybrid_legacy"]


@dataclass
class RetrievalPipelineConfig:
    """检索流水线配置。生产默认 A3；A2 仅消融实验显式开启。"""

    mode: RetrievalMode = "hybrid"
    recall_k: int | None = None
    top_k: int = 3
    use_rerank: bool = False
    legacy_rrf_filter: bool = False
    # 仅消融 A2：允许「宽召回后直接截 Top-K」；生产路径禁止
    allow_wide_without_rerank: bool = False

    def __post_init__(self) -> None:
        if self.recall_k is None:
            if self.use_rerank:
                self.recall_k = config.RETRIEVAL_RECALL_K
            elif self.legacy_rrf_filter or self.mode == "hybrid_legacy":
                self.recall_k = config.SIMILARITY_THRESHOLD
            elif self.allow_wide_without_rerank:
                self.recall_k = config.RETRIEVAL_RECALL_K
            else:
                self.recall_k = config.SIMILARITY_THRESHOLD


def production_pipeline_config() -> RetrievalPipelineConfig:
    """
    生产默认 A3：混合检索宽召回 → Rerank 精排 → Top-N。
    RERANK_ENABLED=false 时回退 A1 hybrid_legacy，不走 A2。
    """
    if config.RERANK_ENABLED:
        return RetrievalPipelineConfig(
            mode="hybrid",
            recall_k=config.RETRIEVAL_RECALL_K,
            top_k=config.RERANK_TOP_N,
            use_rerank=True,
            legacy_rrf_filter=False,
            allow_wide_without_rerank=False,
        )
    return RetrievalPipelineConfig(
        mode="hybrid_legacy",
        recall_k=config.SIMILARITY_THRESHOLD,
        top_k=config.SIMILARITY_THRESHOLD,
        use_rerank=False,
        legacy_rrf_filter=True,
        allow_wide_without_rerank=False,
    )


def recall_candidates(
    query: str,
    vector_service: VectorStoreService,
    *,
    recall_k: int | None = None,
) -> list[Document]:
    """LangGraph 召回阶段：仅宽召回候选，不做最终截断（交给 rerank 节点）。"""
    k = recall_k or config.RETRIEVAL_RECALL_K
    return vector_service.hybrid_search_rrf(
        query,
        recall_k=k,
        top_n=k,
        legacy_filter=False,
    )


def retrieve_documents(
    query: str,
    vector_service: VectorStoreService,
    *,
    pipeline: RetrievalPipelineConfig | None = None,
    reranker: RerankService | None = None,
) -> list[Document]:
    cfg = pipeline or production_pipeline_config()
    reranker = reranker or RerankService()

    if cfg.mode == "dense":
        recall_k = cfg.recall_k or cfg.top_k
        docs = vector_service.search_dense(query, k=recall_k)
    elif cfg.mode == "hybrid_legacy":
        docs = vector_service.hybrid_search_rrf(
            query,
            recall_k=cfg.recall_k or config.SIMILARITY_THRESHOLD,
            top_n=cfg.top_k,
            legacy_filter=True,
        )
        return docs[: cfg.top_k]
    else:
        recall_k = cfg.recall_k or config.RETRIEVAL_RECALL_K
        wide_then_slice = (
            not cfg.use_rerank
            and recall_k > cfg.top_k
            and cfg.allow_wide_without_rerank
        )
        if wide_then_slice:
            # 仅消融 A2
            docs = vector_service.hybrid_search_rrf(
                query,
                recall_k=recall_k,
                top_n=recall_k,
                legacy_filter=False,
            )
            return docs[: cfg.top_k]

        if not cfg.use_rerank and recall_k > cfg.top_k:
            logger.warning(
                "[Retrieval] 已禁止 A2（宽召回无 Rerank），回退 hybrid_legacy (A1)"
            )
            docs = vector_service.hybrid_search_rrf(
                query,
                recall_k=config.SIMILARITY_THRESHOLD,
                top_n=cfg.top_k,
                legacy_filter=True,
            )
            return docs[: cfg.top_k]

        docs = vector_service.hybrid_search_rrf(
            query,
            recall_k=recall_k,
            top_n=recall_k,
            legacy_filter=False,
        )

    if cfg.use_rerank and docs:
        return reranker.rerank(query, docs, top_n=cfg.top_k, enabled=True)

    return docs[: cfg.top_k]
