"""混合检索服务：Dense + Sparse → Merge → Rerank。"""

from __future__ import annotations

from langchain_core.documents import Document

from app.core.vector_stores import VectorStoreService
from app.retrieval.embeddings import get_dense_embeddings


class RetrievalService:
    def __init__(self, vector_service: VectorStoreService | None = None) -> None:
        self._vector = vector_service or VectorStoreService(
            embedding=get_dense_embeddings()
        )

    def hybrid_search(self, query: str) -> tuple[list[str], list[Document]]:
        return self._vector.hybrid_search_workflow(query)

    @staticmethod
    def format_documents(docs: list[Document]) -> str:
        if not docs:
            return "无相关参考资料"
        parts = []
        for i, doc in enumerate(docs):
            score = doc.metadata.get("rerank_score")
            score_hint = f" | 相关度: {score:.3f}" if score is not None else ""
            parts.append(
                f"资料[{i + 1}]: {doc.page_content}\n"
                f"来源: {doc.metadata.get('filename', 'unknown')}{score_hint}"
            )
        return "\n\n".join(parts)
