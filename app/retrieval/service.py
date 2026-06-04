"""混合检索服务（Milvus + BM25 + RRF）。"""

from __future__ import annotations

import os

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from app.core import config_data as config
from app.core.vector_stores import VectorStoreService

os.environ.setdefault("DASHSCOPE_API_KEY", config.DASHSCOPE_API_KEY)


class RetrievalService:
    def __init__(self, vector_service: VectorStoreService | None = None) -> None:
        embedding = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)
        self._vector = vector_service or VectorStoreService(embedding=embedding)

    def hybrid_search(self, query: str) -> tuple[list[str], list[Document]]:
        statuses, docs = self._vector.hybrid_search_workflow(query)
        return statuses, docs

    @staticmethod
    def format_documents(docs: list[Document]) -> str:
        if not docs:
            return "无相关参考资料"
        parts = []
        for i, doc in enumerate(docs):
            parts.append(
                f"资料[{i + 1}]: {doc.page_content}\n"
                f"来源: {doc.metadata.get('filename', 'unknown')}"
            )
        return "\n\n".join(parts)
