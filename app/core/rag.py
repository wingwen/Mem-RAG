"""
RAG 服务：基于 LangGraph StateGraph 的 Agentic RAG 编排。

流程：Router → Memory → Retrieval → Fusion → Generation(stream)
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from langchain_community.embeddings.dashscope import DashScopeEmbeddings

from app.core import config_data as config
from app.core.logger import logger
from app.core.vector_stores import VectorStoreService
from app.graph.nodes.generation import stream_generation
from app.graph.nodes.retrieval import set_retrieval_service
from app.graph.state import AgentState
from app.graph.workflow import get_agent_graph
from app.retrieval.service import RetrievalService

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY


class RagService:
    """LangGraph 驱动的 Agentic RAG 服务。"""

    def __init__(self) -> None:
        embedding = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)
        self.vector_service = VectorStoreService(embedding=embedding)
        set_retrieval_service(RetrievalService(vector_service=self.vector_service))
        self._prep_graph = get_agent_graph(include_generation=False)

    @staticmethod
    def _initial_state(input_text: str, session_id: str) -> AgentState:
        return AgentState(
            query=input_text,
            session_id=session_id,
            rewritten_query="",
            route_decision="",
            memories=[],
            memory_block="",
            history_messages=[],
            active_topic="",
            retrieved_docs=[],
            retrieval_status=[],
            final_context="",
            answer="",
        )

    async def _run_prep_pipeline(self, input_text: str, session_id: str) -> AgentState:
        """执行 Router → Memory → Retrieval → Fusion。"""
        initial = self._initial_state(input_text, session_id)
        return await asyncio.to_thread(self._prep_graph.invoke, initial)

    async def astream_response(self, input_text: str, session_id: str) -> AsyncIterator[str]:
        yield "[状态] LangGraph: Router → Memory → Retrieval → Fusion\n"
        await asyncio.sleep(0)

        state = await self._run_prep_pipeline(input_text, session_id)

        for msg in state.get("retrieval_status") or []:
            if msg:
                yield msg if msg.endswith("\n") else f"{msg}\n"
                await asyncio.sleep(0)

        yield "[状态] 正在启动 Generation 流式生成...\n"
        await asyncio.sleep(0)

        async for chunk in stream_generation(state):
            yield chunk
            await asyncio.sleep(0)

    async def ainvoke(self, input_text: str, session_id: str) -> str:
        """非流式全图执行（含 Generation 节点）。"""
        from app.graph.workflow import build_agent_graph

        graph = build_agent_graph(include_generation=True)
        initial = self._initial_state(input_text, session_id)
        result = await asyncio.to_thread(graph.invoke, initial)
        return result.get("answer") or ""


if __name__ == "__main__":
    async def _demo():
        service = RagService()
        async for part in service.astream_response("海绵宝宝住在哪里？", "demo-session"):
            print(part, end="", flush=True)

    asyncio.run(_demo())
