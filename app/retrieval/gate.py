"""检索命中判定：无有效召回时拒答，避免 LLM 胡编。"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from app.core import config_data as config

# 问句常见虚词，不参与命中判定
_QUERY_STOPWORDS = frozenset({
    "是否", "什么", "怎么", "如何", "为什么", "哪个", "哪些", "哪里", "哪儿", "住在",
    "谁", "多少", "能否", "可以", "是不是", "有没有", "请问", "一下", "告诉",
    "介绍", "关于", "吗", "呢", "啊", "的", "了", "在", "有", "是", "和", "与",
})

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]{2,}")


def extract_query_tokens(query: str) -> list[str]:
    raw = _TOKEN_PATTERN.findall((query or "").lower())
    tokens = []
    for t in raw:
        norm = t.strip()
        if not norm or norm in _QUERY_STOPWORDS:
            continue
        if len(norm) == 2 and norm in _QUERY_STOPWORDS:
            continue
        tokens.append(norm)
    return tokens


def check_retrieval_hit(query: str, docs: list[Document]) -> bool:
    """
    判定检索是否命中知识库。

    - 无文档 / 空正文 → 未命中
    - 严格模式：问句关键词至少一个在 Top 文档中出现 → 命中
    """
    if not config.RETRIEVAL_STRICT_GATE:
        return bool(docs)

    valid_docs = [d for d in docs if (d.page_content or "").strip()]
    if not valid_docs:
        return False

    tokens = extract_query_tokens(query)
    if not tokens:
        return True

    corpus = "\n".join(d.page_content.lower() for d in valid_docs)
    matched = sum(1 for t in tokens if t in corpus)
    min_hits = max(1, config.RETRIEVAL_MIN_KEYWORD_HITS)
    return matched >= min_hits
