"""文本精排：DashScope TextReRank + 关键词回退。"""

from __future__ import annotations

import os
import re

import dashscope
from http import HTTPStatus
from langchain_core.documents import Document

from app.core import config_data as config
from app.core.logger import logger

dashscope.api_key = config.DASHSCOPE_API_KEY


def _doc_key(doc: Document) -> str:
    return f"{doc.metadata.get('filename', '')}:{hash(doc.page_content)}"


def _fallback_rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """API 不可用时的轻量精排：查询词与文档重叠度。"""
    q_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]{2,}", query.lower()))

    def score_doc(doc: Document) -> float:
        text = doc.page_content.lower()
        if not q_tokens:
            return 0.0
        hits = sum(1 for t in q_tokens if t in text)
        return hits / len(q_tokens)

    ranked = sorted(docs, key=score_doc, reverse=True)
    return ranked[:top_n]


class RerankService:
    def rerank(
        self,
        query: str,
        docs: list[Document],
        *,
        top_n: int | None = None,
        enabled: bool | None = None,
    ) -> list[Document]:
        if not docs:
            return []

        top_n = top_n or config.RERANK_TOP_N
        top_n = min(top_n, len(docs))
        use_rerank = config.RERANK_ENABLED if enabled is None else enabled

        if not use_rerank or top_n <= 0:
            return docs[:top_n]

        if not config.DASHSCOPE_API_KEY:
            logger.warning("[Rerank] 未配置 API Key，使用关键词回退精排")
            return _fallback_rerank(query, docs, top_n)

        texts = [d.page_content for d in docs]
        try:
            resp = dashscope.TextReRank.call(
                model=config.RERANK_MODEL,
                query=query,
                documents=texts,
                top_n=top_n,
                return_documents=True,
            )
            if resp.status_code != HTTPStatus.OK:
                logger.warning(
                    f"[Rerank] API 失败 HTTP {resp.status_code}: "
                    f"{getattr(resp, 'message', resp)}，回退关键词精排"
                )
                return _fallback_rerank(query, docs, top_n)

            results = resp.output.get("results", []) if resp.output else []
            if not results:
                return _fallback_rerank(query, docs, top_n)

            key_to_doc = {_doc_key(d): d for d in docs}
            reranked: list[Document] = []
            seen: set[str] = set()

            for item in results:
                idx = item.get("index")
                if idx is not None and 0 <= idx < len(docs):
                    doc = docs[idx]
                else:
                    text = ""
                    if isinstance(item.get("document"), dict):
                        text = item["document"].get("text", "")
                    elif isinstance(item.get("document"), str):
                        text = item["document"]
                    doc = next((d for d in docs if d.page_content == text), None)
                    if doc is None:
                        continue

                key = _doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                doc.metadata["rerank_score"] = item.get("relevance_score", 0)
                reranked.append(doc)

            if reranked:
                logger.info(f"[Rerank] {config.RERANK_MODEL} 精排 Top-{len(reranked)}")
                return reranked[:top_n]

        except Exception as exc:
            logger.warning(f"[Rerank] 调用异常: {exc}，回退关键词精排")

        return _fallback_rerank(query, docs, top_n)
