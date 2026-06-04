"""BGE-Reranker-v2 精排。"""

from __future__ import annotations

from langchain_core.documents import Document

from app.core import config_data as config
from app.core.logger import logger

_reranker = None
_reranker_unavailable = False


def get_reranker():
    global _reranker, _reranker_unavailable
    if _reranker_unavailable:
        return None
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker

            logger.info(f"[Reranker] 加载 {config.RERANKER_MODEL_NAME}")
            _reranker = FlagReranker(config.RERANKER_MODEL_NAME, use_fp16=True)
        except Exception as exc:
            logger.warning(f"[Reranker] 不可用，将跳过精排: {exc}")
            _reranker_unavailable = True
            return None
    return _reranker


def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int | None = None,
) -> list[Document]:
    """对候选文档精排，降低语义误召回。"""
    if not docs:
        return []

    k = top_k or config.RETRIEVAL_TOP_K
    if len(docs) <= k and not config.RERANKER_ENABLED:
        return docs[:k]

    if not config.RERANKER_ENABLED:
        return docs[:k]

    reranker = get_reranker()
    if reranker is None:
        return docs[:k]

    pairs = [[query, doc.page_content] for doc in docs]
    try:
        scores = reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
    except Exception as exc:
        logger.warning(f"[Reranker] 精排失败，回退 RRF 顺序: {exc}")
        return docs[:k]

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    result: list[Document] = []
    for doc, score in ranked[:k]:
        doc.metadata = {**doc.metadata, "rerank_score": float(score)}
        result.append(doc)
    return result
