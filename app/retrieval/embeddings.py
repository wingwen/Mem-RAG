"""Dense Embedding 工厂：DashScope / BGE-M3。"""

from __future__ import annotations

from typing import Protocol

from app.core import config_data as config
from app.core.logger import logger


class DenseEmbeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


_bge_m3_wrapper: "BGEM3Embeddings | None" = None


class BGEM3Embeddings:
    """BGE-M3 稠密向量（FlagEmbedding）。"""

    def __init__(self, model_name: str | None = None) -> None:
        from FlagEmbedding import BGEM3FlagModel

        name = model_name or config.DENSE_MODEL_NAME
        logger.info(f"[Embedding] 加载 BGE-M3: {name}")
        self._model = BGEM3FlagModel(name, use_fp16=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        output = self._model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            max_length=8192,
        )
        vecs = output["dense_vecs"]
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_dense_embeddings() -> DenseEmbeddings:
    provider = (config.DENSE_EMBEDDING_PROVIDER or "dashscope").lower()
    if provider == "bge-m3":
        global _bge_m3_wrapper
        if _bge_m3_wrapper is None:
            _bge_m3_wrapper = BGEM3Embeddings()
        return _bge_m3_wrapper

    import os

    from langchain_community.embeddings import DashScopeEmbeddings

    os.environ.setdefault("DASHSCOPE_API_KEY", config.DASHSCOPE_API_KEY)
    logger.info(f"[Embedding] 使用 DashScope: {config.EMBEDDINGS_MODEL}")
    return DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)
