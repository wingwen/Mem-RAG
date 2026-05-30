import asyncio
import os
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from app.core.vector_stores import VectorStoreService  # 确保文件名匹配
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from app.core import config_data as config
from app.core.prompts import rag_prompt_template
from langchain_community.chat_models.tongyi import ChatTongyi
from app.core.structured_memory import get_structured_memory
from app.utils.file_history_store import get_history
from app.core.logger import logger

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY


class RagService(object):
    def __init__(self) -> None:
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)
        )
        self.prompt_template = rag_prompt_template
        self.chat_model = ChatTongyi(model="qwen-max", streaming=True)
        self.chain = self.__get_chain()

    @staticmethod
    def _format_document(docs: list[Document]) -> str:
        if not docs:
            return "无相关参考资料"
        formatted_docs = []
        for i, doc in enumerate(docs):
            content = f"资料[{i + 1}]: {doc.page_content}\n来源: {doc.metadata.get('filename')}"
            formatted_docs.append(content)
        return "\n\n".join(formatted_docs)

    def _search_documents(self, query: str) -> list[Document]:
        retriever = self.vector_service.get_retriever()
        return retriever.invoke(query)

    @staticmethod
    def _extract_chunk_content(chunk) -> str:
        content = chunk.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content) if content else ""

    async def astream_response(self, input_text: str, session_id: str) -> AsyncIterator[str]:
        """先流式输出检索状态，再逐 token 流式输出模型回答。"""
        retrieval_statuses = [
            "[状态] 正在加载检索器...\n",
            "[状态] 正在加载向量检索器...\n",
            "[状态] 正在加载 BM25 检索器...\n",
            "[状态] 正在启动 RRF 倒数秩融合策略...\n",
            "[状态] 正在执行混合搜索...\n",
        ]
        for message in retrieval_statuses:
            yield message
            await asyncio.sleep(0)

        docs = await asyncio.to_thread(self._search_documents, input_text)

        post_statuses = [
            "[状态] 搜索完成，正在处理结果...\n",
            "[状态] 正在准备大模型回答...\n",
        ]
        for message in post_statuses:
            yield message
            await asyncio.sleep(0)

        context = self._format_document(docs)
        memory_ctx = get_structured_memory().build_context(session_id, input_text)
        prompt_messages = self.prompt_template.format_messages(
            memory_context=memory_ctx.memory_block,
            context=context,
            input=input_text,
            history=memory_ctx.history_messages,
        )

        async for chunk in self.chat_model.astream(prompt_messages):
            content = self._extract_chunk_content(chunk)
            if content:
                yield content
                await asyncio.sleep(0)

    def __get_chain(self):
        """获取最终的执行链"""

        def retrieve_context(input_data: dict):
            query = input_data["input"]
            _, docs = self.vector_service.hybrid_search_workflow(query)
            return self._format_document(docs)

        # 构建处理链
        # 注意：RunnableWithMessageHistory 会传入整个 dict {"input": "...", "history": [...]}
        chain = (
                {
                    "context": RunnableLambda(retrieve_context),
                    "input": lambda x: x["input"],
                    "history": lambda x: x["history"]
                }
                | self.prompt_template
                | self.chat_model
                | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain


if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "user002"
        }
    }

    # 第一次运行建议使用包含关键词的问题测试 RAG 效果
    service = RagService()
    res = service.chain.invoke({"input": "周杰伦出生在什么时候？"}, session_config)

    logger.info("-" * 30)
    logger.info(f"AI 回复: {res}")