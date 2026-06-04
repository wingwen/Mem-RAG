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
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3-8b")
CHAT_MODEL_LIGHT = os.getenv("CHAT_MODEL_LIGHT", "qwen3-8b")

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

# 生产检索 A3：宽召回 + Rerank（A2 宽召回无 Rerank 仅消融可用，生产已禁用）
RETRIEVAL_RECALL_K = int(os.getenv("RETRIEVAL_RECALL_K", "15"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

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
