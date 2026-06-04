"""BM25 稀疏检索。"""

from __future__ import annotations

import os
import pickle

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.core import config_data as config


def load_bm25_documents() -> list[Document]:
    if not os.path.exists(config.BM25_CORPUS_PATH):
        return []
    with open(config.BM25_CORPUS_PATH, "rb") as f:
        corpus = pickle.load(f)
    if not corpus:
        return []

    documents: list[Document] = []
    for item in corpus:
        if isinstance(item, dict):
            documents.append(
                Document(
                    page_content=item.get("text", ""),
                    metadata={
                        "filename": item.get("filename", "unknown"),
                        "source": "bm25",
                    },
                )
            )
        else:
            documents.append(
                Document(
                    page_content=str(item),
                    metadata={"filename": "unknown", "source": "bm25"},
                )
            )
    return documents


def sparse_search(query: str, k: int | None = None) -> list[Document]:
    docs = load_bm25_documents()
    if not docs:
        return []
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k or config.RETRIEVAL_CANDIDATE_K
    results = retriever.invoke(query)
    for doc in results:
        doc.metadata = {**doc.metadata, "source": "bm25"}
    return results
