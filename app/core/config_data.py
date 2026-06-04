# config_data.py
import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT.parent / ".env")
load_dotenv(_PROJECT_ROOT / ".env")

# 基础配置
md5_path = os.getenv("MD5_PATH", "./database/md5.text")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-v4")

# LLM（DashScope 通义千问）
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-max")
CHAT_MODEL_LIGHT = os.getenv("CHAT_MODEL_LIGHT", "qwen-turbo")
# Query Rewrite 默认补全的产品/项目名（指代消解）
REWRITE_PROJECT_NAME = os.getenv("REWRITE_PROJECT_NAME", "Mem-RAG")

# Milvus 配置 (使用 Milvus Lite 本地文件模式)
MILVUS_URI = os.getenv("MILVUS_URI", "./database/milvus_db.db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")

# 语义分割配置
BREAKPOINT_TYPE = os.getenv("BREAKPOINT_TYPE", "percentile")
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "1"))
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "hybrid")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))
SEMANTIC_MAX_CHARS = int(os.getenv("SEMANTIC_MAX_CHARS", "30000"))

# 混合检索与召回配置
BM25_CORPUS_PATH = os.getenv("BM25_CORPUS_PATH", "./database/bm25_corpus.pkl")
SIMILARITY_THRESHOLD = int(os.getenv("SIMILARITY_THRESHOLD", "3"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.7"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.3"))

# Hybrid Retrieval: Dense (Milvus) + Sparse (BM25) → Merge (RRF) → Rerank
DENSE_EMBEDDING_PROVIDER = os.getenv("DENSE_EMBEDDING_PROVIDER", "dashscope")  # dashscope | bge-m3
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-m3")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2")
HYBRID_MERGE_STRATEGY = os.getenv("HYBRID_MERGE_STRATEGY", "rrf")
RRF_K = int(os.getenv("RRF_K", "60"))
RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "10"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", str(SIMILARITY_THRESHOLD)))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "12"))

# 检索拒答门控（无命中不调用 LLM，避免胡编）
RETRIEVAL_STRICT_GATE = os.getenv("RETRIEVAL_STRICT_GATE", "true").lower() == "true"
RETRIEVAL_MIN_KEYWORD_HITS = int(os.getenv("RETRIEVAL_MIN_KEYWORD_HITS", "1"))
NO_KB_ANSWER_MESSAGE = os.getenv(
    "NO_KB_ANSWER_MESSAGE",
    "知识库中暂无相关信息，请尝试上传相关文档或换个问法。",
)

# 文本限制
MAX_SPLIT_CHAR_NUMBER = int(os.getenv("MAX_SPLIT_CHAR_NUMBER", "1000"))

# 结构化记忆配置
MEMORY_MAX_RECENT_TURNS = int(os.getenv("MEMORY_MAX_RECENT_TURNS", "2"))
MEMORY_SAME_TOPIC_TURNS = int(os.getenv("MEMORY_SAME_TOPIC_TURNS", "4"))
MEMORY_MAX_TOPICS = int(os.getenv("MEMORY_MAX_TOPICS", "8"))
MEMORY_TOPIC_MATCH_THRESHOLD = float(os.getenv("MEMORY_TOPIC_MATCH_THRESHOLD", "0.35"))

# API 服务
ASYNC_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+aiomysql://root:your_password@localhost:3306/memrag_db",
)
SALT_SUFFIX = os.getenv("SALT_SUFFIX", "MYRAG")

# 安全 / CORS
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]
# 开发时允许 IDE 预览（如 PyCharm 63342）等 localhost 任意端口跨域
CORS_ALLOW_LOCALHOST_ANY_PORT = os.getenv(
    "CORS_ALLOW_LOCALHOST_ANY_PORT", "true"
).lower() == "true"
CORS_ORIGIN_REGEX = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"http://(localhost|127\.0\.0\.1)(:\d+)?",
)
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")
