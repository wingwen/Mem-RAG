import os
import pickle
import hashlib
import time
from datetime import datetime

from pymilvus import connections, MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
from app.core import config_data as config
from app.core.document_parser import parse_document, SUPPORTED_EXTENSIONS
from app.core.text_splitter import HybridTextSplitter
from app.core.logger import logger

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY


def get_string_md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def check_md5(md5_str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding="utf-8").close()
        return False
    with open(config.md5_path, 'r', encoding="utf-8") as f:
        return md5_str in [line.strip() for line in f.readlines()]


def save_md5(md5):
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5 + '\n')


class KnowledgeBaseService:
    def __init__(self):
        logger.info("[System] 初始化 KnowledgeBaseService...")
        self.embeddings = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)

        logger.info(f"[Milvus] 正在启动本地引擎: {config.MILVUS_URI}")
        try:
            connections.connect(alias="default", uri=config.MILVUS_URI)
            time.sleep(2)
            logger.info("[Milvus] 引擎就绪。")
        except Exception as e:
            logger.error(f"[Error] 引擎启动失败: {e}")

        logger.info(f"[Splitter] 加载分块器 (strategy={config.CHUNK_STRATEGY})...")
        self.splitter = HybridTextSplitter(self.embeddings)
        self.bm25_corpus = self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        if os.path.exists(config.BM25_CORPUS_PATH):
            try:
                with open(config.BM25_CORPUS_PATH, 'rb') as f:
                    raw = pickle.load(f)
                return [self._normalize_corpus_item(item) for item in raw]
            except (EOFError, pickle.UnpicklingError) as e:
                logger.error(f"[Error] 加载 BM25 语料库失败: {e}")
                logger.info("[BM25] 使用空语料库")
                return []
        return []

    @staticmethod
    def _normalize_corpus_item(item):
        if isinstance(item, dict):
            return {
                "text": item.get("text", ""),
                "filename": item.get("filename", "unknown"),
            }
        return {"text": str(item), "filename": "unknown"}

    def _save_bm25_corpus(self):
        os.makedirs(os.path.dirname(config.BM25_CORPUS_PATH), exist_ok=True)
        with open(config.BM25_CORPUS_PATH, 'wb') as f:
            pickle.dump(self.bm25_corpus, f)

    def upload_by_bytes(self, file_bytes: bytes, filename: str) -> str:
        """解析 PDF/Word/TXT 文件并入库。"""
        logger.info(f"[Process] 开始解析文件: {filename}")
        try:
            text = parse_document(file_bytes, filename)
        except (ValueError, ImportError) as exc:
            logger.error(f"[Error] 文件解析失败: {exc}")
            return f"【失败】{exc}"
        return self.upload_by_str(text, filename)

    def upload_by_str(self, data, filename):
        logger.info(f"\n[Process] 开始处理文件: {filename}")
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "【跳过】内容已在库中"

        logger.info(f"[Split] 执行 {config.CHUNK_STRATEGY} 分块策略...")
        knowledge_chunks = self.splitter.split_text(data)
        if not knowledge_chunks:
            return "【失败】分块结果为空，请检查文档内容"

        logger.info(f"[Split] 共生成 {len(knowledge_chunks)} 个文本块")

        logger.info("[Storage] 正在通过底层 Client 写入 Milvus...")
        try:
            client = MilvusClient(config.MILVUS_URI)

            logger.info("[Storage] 正在生成向量并自动获取维度...")
            vectors = self.embeddings.embed_documents(knowledge_chunks)
            actual_dim = len(vectors[0])
            logger.info(f"[Storage] 检测到模型输出维度为: {actual_dim}")

            if not client.has_collection(config.COLLECTION_NAME):
                logger.info(f"[Storage] 创建集合: {config.COLLECTION_NAME}")
                client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    dimension=actual_dim,
                    auto_id=True,
                    enable_dynamic_field=True
                )

            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_to_insert = [
                {
                    "vector": v,
                    "text": t,
                    "filename": filename,
                    "create_time": cur_time
                }
                for v, t in zip(vectors, knowledge_chunks)
            ]

            client.insert(collection_name=config.COLLECTION_NAME, data=data_to_insert)
            client.load_collection(collection_name=config.COLLECTION_NAME)
            client.close()
            logger.info(f"[Storage] 成功存入 {len(data_to_insert)} 条数据！")

        except Exception as e:
            logger.error(f"[Error] 写入失败: {e}")
            return f"【失败】{e}"

        self.bm25_corpus.extend(
            [{"text": chunk, "filename": filename} for chunk in knowledge_chunks]
        )
        self._save_bm25_corpus()
        save_md5(md5_hex)
        return f"【成功】内容已载入数据库，共 {len(knowledge_chunks)} 个分块"

    @staticmethod
    def supported_formats() -> list[str]:
        return sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)


if __name__ == '__main__':
    service = KnowledgeBaseService()
    test_text = "周杰伦出生于1979年，代表作有《青花瓷》。"
    logger.info(service.upload_by_str(test_text, "jay_chou_test"))
