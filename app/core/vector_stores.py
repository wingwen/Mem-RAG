"""Milvus 向量服务（兼容层，内部委托 HybridRetriever）。"""

from __future__ import annotations

import os

from langchain_core.documents import Document

from app.core import config_data as config
from app.core.logger import logger
from app.retrieval.embeddings import DenseEmbeddings, get_dense_embeddings
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.dense import MilvusDenseStore

os.environ.setdefault("DASHSCOPE_API_KEY", config.DASHSCOPE_API_KEY)


class VectorStoreService:
    """对外保留原接口；检索走 Dense+Sparse→Merge→Rerank 管道。"""

    def __init__(self, embedding: DenseEmbeddings | None = None) -> None:
        logger.info("[Retriever] 初始化 Hybrid Retrieval 服务...")
        self.embedding = embedding or get_dense_embeddings()
        self._dense_store = MilvusDenseStore(self.embedding)
        self._hybrid = HybridRetriever(
            embedding=self.embedding,
            dense_store=self._dense_store,
        )

    def search_milvus(self, query: str, k: int | None = None) -> list[Document]:
        return self._dense_store.search(query, k=k or config.RETRIEVAL_CANDIDATE_K)

    def hybrid_search_workflow(self, query: str) -> tuple[list[str], list[Document]]:
        return self._hybrid.search(query)

    def get_retriever(self):
        """兼容旧代码：返回可调 invoke 的检索器。"""
        from langchain_core.retrievers import BaseRetriever

        hybrid = self._hybrid

        class HybridRetrieverWrapper(BaseRetriever):
            def _get_relevant_documents(self, query: str) -> list[Document]:
                _, docs = hybrid.search(query)
                return docs

        return HybridRetrieverWrapper()


if __name__ == "__main__":
    service = VectorStoreService()
    query = "海绵宝宝住在哪里"
    logger.info(f"\n[Search] 执行查询: {query}")
    statuses, results = service.hybrid_search_workflow(query)
    for s in statuses:
        print(s, end="")
    for i, doc in enumerate(results):
        logger.info(
            f"结果 {i + 1}: {doc.page_content[:80]}... "
            f"(来源: {doc.metadata.get('filename')}, rerank={doc.metadata.get('rerank_score')})"
        )
