"""混合分块策略：语义分块 + 递归分块 + 大小约束。"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core import config_data as config
from app.core.logger import logger


class HybridTextSplitter:
    """
    分块流程：
    1. 短文本直接返回
    2. 超长文本或 strategy=recursive：递归字符分块
    3. hybrid：语义分块后，超大块二次切分、过小块合并
    """

    def __init__(self, embeddings):
        self.max_chunk = config.MAX_SPLIT_CHAR_NUMBER
        self.min_chunk = config.MIN_CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.strategy = config.CHUNK_STRATEGY
        self.semantic_max_chars = config.SEMANTIC_MAX_CHARS

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " ", ""],
            length_function=len,
        )
        self.semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=config.BREAKPOINT_TYPE,
            buffer_size=config.BUFFER_SIZE,
        )

    def split_text(self, text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        if len(cleaned) <= self.min_chunk:
            return [cleaned]

        if self.strategy == "recursive" or len(cleaned) > self.semantic_max_chars:
            if len(cleaned) > self.semantic_max_chars:
                logger.info(
                    f"[Splitter] 文本长度 {len(cleaned)} 超过语义分块上限 "
                    f"{self.semantic_max_chars}，使用递归分块"
                )
            chunks = self.recursive_splitter.split_text(cleaned)
            return self._merge_small_chunks(chunks)

        if self.strategy == "semantic":
            chunks = self._semantic_split(cleaned)
            return self._enforce_size_constraints(chunks)

        # hybrid（默认）
        chunks = self._semantic_split(cleaned)
        chunks = self._enforce_size_constraints(chunks)
        return self._merge_small_chunks(chunks)

    def _semantic_split(self, text: str) -> list[str]:
        try:
            return self.semantic_splitter.split_text(text)
        except Exception as exc:
            logger.warning(f"[Splitter] 语义分块失败，回退递归分块: {exc}")
            return self.recursive_splitter.split_text(text)

    def _enforce_size_constraints(self, chunks: list[str]) -> list[str]:
        result = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if len(chunk) <= self.max_chunk:
                result.append(chunk)
            else:
                result.extend(self.recursive_splitter.split_text(chunk))
        return result

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return []

        merged: list[str] = []
        buffer = ""

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if not buffer:
                buffer = chunk
                continue

            if len(buffer) < self.min_chunk and len(buffer) + len(chunk) + 2 <= self.max_chunk:
                buffer = f"{buffer}\n\n{chunk}"
            else:
                merged.append(buffer)
                buffer = chunk

        if buffer:
            if merged and len(buffer) < self.min_chunk and len(merged[-1]) + len(buffer) + 2 <= self.max_chunk:
                merged[-1] = f"{merged[-1]}\n\n{buffer}"
            else:
                merged.append(buffer)

        return merged
