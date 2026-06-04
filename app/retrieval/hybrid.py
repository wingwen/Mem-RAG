"""
Hybrid Retriever：Dense (Milvus) + Sparse (BM25) → Merge (RRF) → Rerank (BGE-Reranker-v2)
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.core import config_data as config
from app.core.logger import logger
from app.retrieval.dense import MilvusDenseStore
from app.retrieval.embeddings import DenseEmbeddings, get_dense_embeddings
from app.retrieval.reranker import rerank_documents
from app.retrieval.sparse import sparse_search


def _doc_key(doc: Document) -> str:
    return f"{doc.metadata.get('filename', '')}::{doc.page_content[:200]}"


def rrf_merge(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    *,
    rrf_k: int | None = None,
    limit: int | None = None,
) -> list[Document]:
    """Reciprocal Rank Fusion 合并双路召回。"""
    k = rrf_k or config.RRF_K
    cap = limit or config.RETRIEVAL_CANDIDATE_K
    scores: dict[str, dict] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        key = _doc_key(doc)
        if key not in scores:
            scores[key] = {"doc": doc, "score": 0.0}
        scores[key]["score"] += 1.0 / (rank + k)
        scores[key]["doc"].metadata = {**scores[key]["doc"].metadata, "dense_rank": rank}

    for rank, doc in enumerate(sparse_docs, start=1):
        key = _doc_key(doc)
        if key not in scores:
            scores[key] = {"doc": doc, "score": 0.0}
        scores[key]["score"] += 1.0 / (rank + k)
        scores[key]["doc"].metadata = {**scores[key]["doc"].metadata, "sparse_rank": rank}

    merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    results: list[Document] = []
    for item in merged[:cap]:
        doc = item["doc"]
        doc.metadata = {**doc.metadata, "rrf_score": item["score"]}
        results.append(doc)
    return results


class HybridRetriever:
    """
    Query → Dense Search + Sparse Search → Merge → Rerank → Top-K
    """

    def __init__(
        self,
        embedding: DenseEmbeddings | None = None,
        dense_store: MilvusDenseStore | None = None,
    ) -> None:
        self.embedding = embedding or get_dense_embeddings()
        self.dense_store = dense_store or MilvusDenseStore(self.embedding)

    def search(self, query: str) -> tuple[list[str], list[Document]]:
        statuses: list[str] = []
        provider = config.DENSE_EMBEDDING_PROVIDER
        statuses.append(f"[状态] Hybrid Retriever 启动 (Dense={provider}, Sparse=BM25)\n")

        statuses.append("[状态] Dense Search (Milvus)...\n")
        dense_docs = self.dense_store.search(query)
        statuses.append(f"[状态] Dense 召回 {len(dense_docs)} 条\n")

        statuses.append("[状态] Sparse Search (BM25)...\n")
        sparse_docs = sparse_search(query)
        statuses.append(f"[状态] Sparse 召回 {len(sparse_docs)} 条\n")

        if not dense_docs and not sparse_docs:
            statuses.append("[状态] 双路均无结果\n")
            return statuses, []

        statuses.append(f"[状态] Merge ({config.HYBRID_MERGE_STRATEGY.upper()})...\n")
        if sparse_docs:
            merged = rrf_merge(dense_docs, sparse_docs)
        else:
            merged = dense_docs[: config.RETRIEVAL_CANDIDATE_K]
            statuses.append("[状态] BM25 语料为空，仅使用 Dense 结果\n")
        statuses.append(f"[状态] Merge 候选 {len(merged)} 条\n")

        if config.RERANKER_ENABLED and merged:
            statuses.append(f"[状态] Rerank ({config.RERANKER_MODEL_NAME})...\n")
            final_docs = rerank_documents(query, merged)
            statuses.append(f"[状态] Rerank 完成，Top-{len(final_docs)} 条\n")
        else:
            final_docs = merged[: config.RETRIEVAL_TOP_K]
            statuses.append(f"[状态] 跳过 Rerank，取 Top-{len(final_docs)} 条\n")

        statuses.append("[状态] Hybrid Retrieval 完成\n")
        return statuses, final_docs
