import os
import pickle
import time

from langchain_core.documents import Document
from pymilvus import MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  # 使用你确认正确的导入

from app.core import config_data as config
from app.core.logger import logger

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY


class VectorStoreService:
    def __init__(self, embedding):
        logger.info("[Retriever] 初始化检索服务...")
        self.embedding = embedding
        # 建立底层 Client，配置 gRPC 选项以避免 too_many_pings 错误
        try:
            # 配置 gRPC 选项，调整 keepalive 时间和频率
            grpc_options = {
                'grpc.keepalive_time_ms': 30000,  # 30秒发送一次 keepalive ping
                'grpc.keepalive_timeout_ms': 10000,  # 10秒超时
                'grpc.keepalive_permit_without_calls': True,  # 允许在没有调用时发送 keepalive
                'grpc.http2.max_pings_without_data': 5,  # 最大无数据 ping 次数
                'grpc.http2.min_time_between_pings_ms': 5000  # 两次 ping 之间的最小时间
            }
            
            self.client = MilvusClient(
                config.MILVUS_URI,
                grpc_options=grpc_options
            )
            logger.info("[Milvus] 成功连接到数据库")
        except Exception as e:
            logger.error(f"[Error] 连接 Milvus 失败: {e}")
            logger.info("[Milvus] 尝试创建新的数据库...")
            # 尝试创建新的数据库
            try:
                # 确保 database 目录存在
                os.makedirs(os.path.dirname(config.MILVUS_URI), exist_ok=True)
                # 同样使用 gRPC 选项创建新数据库
                grpc_options = {
                    'grpc.keepalive_time_ms': 30000,
                    'grpc.keepalive_timeout_ms': 10000,
                    'grpc.keepalive_permit_without_calls': True,
                    'grpc.http2.max_pings_without_data': 5,
                    'grpc.http2.min_time_between_pings_ms': 5000
                }
                self.client = MilvusClient(
                    config.MILVUS_URI,
                    grpc_options=grpc_options
                )
                logger.info("[Milvus] 成功创建并连接到新数据库")
            except Exception as e2:
                logger.error(f"[Error] 创建 Milvus 数据库失败: {e2}")
                raise

        self._ensure_collection_loaded()

    def _ensure_collection_loaded(self) -> bool:
        """Milvus Lite 集合处于 released 状态时需先 load 才能 search。"""
        if not self.client.has_collection(config.COLLECTION_NAME):
            return False
        try:
            self.client.load_collection(collection_name=config.COLLECTION_NAME)
            return True
        except Exception as e:
            logger.error(f"[Error] 加载 Milvus 集合失败: {e}")
            return False

    def search_milvus(self, query, k=3):
        """底层手动搜索，绕过 LangChain Milvus 类的 Bug"""
        if not self.client.has_collection(config.COLLECTION_NAME):
            logger.info(f"[Milvus] 集合 {config.COLLECTION_NAME} 不存在，返回空结果")
            return []

        if not self._ensure_collection_loaded():
            return []
        # 1. 生成查询向量
        query_vector = self.embedding.embed_query(query)

        # 2. 执行搜索
        try:
            res = self.client.search(
                collection_name=config.COLLECTION_NAME,
                data=[query_vector],
                limit=k,
                output_fields=["text", "filename"]  # 必须匹配 knowledge_base 存入的字段
            )

            # 3. 将结果转为 LangChain 的 Document 格式
            docs = []
            for hit in res[0]:
                doc = Document(
                    page_content=hit['entity']['text'],
                    metadata={"filename": hit['entity']['filename'], "score": hit['distance']}
                )
                docs.append(doc)

            return docs
        except Exception as e:
            logger.error(f"[Error] 搜索 Milvus 失败: {e}")
            return []

    def search_dense(self, query: str, k: int) -> list[Document]:
        return self.search_milvus(query, k=k)

    def hybrid_search_rrf(
        self,
        query: str,
        *,
        recall_k: int,
        top_n: int,
        legacy_filter: bool = False,
        rrf_k: int = 60,
    ) -> list[Document]:
        """Milvus + BM25 RRF 融合。legacy_filter=True 为旧版 97% 阈值截断。"""
        dense_docs = self.search_milvus(query, k=recall_k)
        sparse_retriever = self._get_bm25_retriever()
        if not sparse_retriever:
            return dense_docs[:top_n]

        sparse_retriever.k = recall_k
        sparse_docs = sparse_retriever.invoke(query)

        all_results: list[tuple[Document, int, int]] = []
        for rank, doc in enumerate(dense_docs, 1):
            all_results.append((doc, 0, rank))
        for rank, doc in enumerate(sparse_docs, 1):
            all_results.append((doc, 1, rank))

        doc_scores: dict[str, dict] = {}
        for doc, _retriever_idx, rank in all_results:
            doc_id = str(hash(doc.page_content))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0.0}
            doc_scores[doc_id]["score"] += 1.0 / (rank + rrf_k)

        sorted_items = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        if not sorted_items:
            return []

        if legacy_filter:
            max_score = sorted_items[0]["score"]
            relevant_docs: list[Document] = []
            for item in sorted_items:
                normalized = (item["score"] / max_score) * 100 if max_score else 0
                if normalized >= 97:
                    item["doc"].metadata["rrf_score"] = item["score"]
                    relevant_docs.append(item["doc"])
                    if len(relevant_docs) >= top_n:
                        break
            return relevant_docs

        for item in sorted_items:
            item["doc"].metadata["rrf_score"] = item["score"]
        return [item["doc"] for item in sorted_items[:top_n]]

    def get_retriever(self):
        """LangChain Retriever 兼容（评估脚本等）。"""
        from langchain_core.retrievers import BaseRetriever
        from typing import List

        service = self

        class PipelineRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str) -> List[Document]:
                from app.retrieval.pipeline import (
                    production_pipeline_config,
                    retrieve_documents,
                )

                return retrieve_documents(
                    query, service, pipeline=production_pipeline_config()
                )

        return PipelineRetriever()

    def _get_bm25_retriever(self):
        if os.path.exists(config.BM25_CORPUS_PATH):
            with open(config.BM25_CORPUS_PATH, 'rb') as f:
                corpus = pickle.load(f)
            if corpus:
                documents = []
                for item in corpus:
                    if isinstance(item, dict):
                        documents.append(Document(
                            page_content=item.get("text", ""),
                            metadata={"filename": item.get("filename", "unknown")},
                        ))
                    else:
                        documents.append(Document(
                            page_content=str(item),
                            metadata={"filename": "unknown"},
                        ))
                r = BM25Retriever.from_documents(documents)
                r.k = config.SIMILARITY_THRESHOLD
                return r
        return None

    def hybrid_search_workflow(self, query: str, *, use_rerank: bool | None = None):
        """
        生产检索工作流（默认 A3）：宽召回 RRF → Rerank；关闭 Rerank 时用 A1 legacy。
        """
        from app.retrieval.pipeline import production_pipeline_config, retrieve_documents

        cfg = production_pipeline_config()
        if use_rerank is not None:
            cfg.use_rerank = use_rerank

        status_messages = [
            "[状态] 正在加载检索器...\n",
            "[状态] 正在加载向量检索器...\n",
            "[状态] 正在加载 BM25 检索器...\n",
            "[状态] 正在启动 RRF 倒数秩融合策略...\n",
        ]
        if cfg.use_rerank:
            status_messages.append(
                f"[状态] A3 宽召回 Top-{config.RETRIEVAL_RECALL_K} → "
                f"Rerank ({config.RERANK_MODEL}) Top-{cfg.top_k}...\n"
            )
        else:
            status_messages.append(
                f"[状态] A1 hybrid_legacy Top-{cfg.top_k}（未启用宽召回无精排）...\n"
            )

        results = retrieve_documents(query, self, pipeline=cfg)
        status_messages.append("[状态] 检索完成，正在处理结果...\n")
        return status_messages, results


if __name__ == "__main__":
    embeddings = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)
    service = VectorStoreService(embeddings)
    retriever = service.get_retriever()

    query = "周杰伦出生在什么时候？"
    logger.info(f"\n[Search] 执行查询: {query}")

    try:
        results = retriever.invoke(query)
        for i, doc in enumerate(results):
            logger.info(f"结果 {i + 1}: {doc.page_content} (来源: {doc.metadata.get('filename')})")
    except Exception as e:
        logger.error(f"检索出错: {e}")