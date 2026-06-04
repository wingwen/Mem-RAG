"""Milvus 稠密检索。"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from pymilvus import MilvusClient

from app.core import config_data as config
from app.core.logger import logger
from app.retrieval.embeddings import DenseEmbeddings, get_dense_embeddings

_GRPC_OPTIONS = {
    "grpc.keepalive_time_ms": 30000,
    "grpc.keepalive_timeout_ms": 10000,
    "grpc.keepalive_permit_without_calls": True,
    "grpc.http2.max_pings_without_data": 5,
    "grpc.http2.min_time_between_pings_ms": 5000,
}


class MilvusDenseStore:
    def __init__(self, embedding: DenseEmbeddings | None = None) -> None:
        self.embedding = embedding or get_dense_embeddings()
        os.makedirs(os.path.dirname(config.MILVUS_URI) or ".", exist_ok=True)
        self.client = MilvusClient(config.MILVUS_URI, grpc_options=_GRPC_OPTIONS)
        logger.info("[Milvus] Dense Store 已连接")
        self._ensure_collection_loaded()

    def _ensure_collection_loaded(self) -> bool:
        if not self.client.has_collection(config.COLLECTION_NAME):
            return False
        try:
            self.client.load_collection(collection_name=config.COLLECTION_NAME)
            return True
        except Exception as exc:
            logger.error(f"[Milvus] 加载集合失败: {exc}")
            return False

    def search(self, query: str, k: int | None = None) -> list[Document]:
        limit = k or config.RETRIEVAL_CANDIDATE_K
        if not self.client.has_collection(config.COLLECTION_NAME):
            logger.info(f"[Milvus] 集合 {config.COLLECTION_NAME} 不存在")
            return []
        if not self._ensure_collection_loaded():
            return []

        query_vector = self.embedding.embed_query(query)
        try:
            res = self.client.search(
                collection_name=config.COLLECTION_NAME,
                data=[query_vector],
                limit=limit,
                output_fields=["text", "filename"],
            )
        except Exception as exc:
            logger.error(f"[Milvus] Dense Search 失败: {exc}")
            return []

        docs: list[Document] = []
        for hit in res[0]:
            docs.append(
                Document(
                    page_content=hit["entity"]["text"],
                    metadata={
                        "filename": hit["entity"]["filename"],
                        "milvus_distance": hit["distance"],
                        "source": "milvus",
                    },
                )
            )
        return docs
