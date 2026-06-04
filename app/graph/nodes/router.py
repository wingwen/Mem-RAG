"""Router：查询改写与路由决策。"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from app.core.logger import logger
from app.graph.state import AgentState
from app.llm.factory import get_light_chat_model

_VAGUE_PATTERN = re.compile(
    r"^(它|他|她|这个|那个|继续|还有呢|然后呢|怎么说|为什么|怎么样)[\?？。！]?$",
    re.I,
)


def router_node(state: AgentState) -> dict:
    query = (state.get("query") or "").strip()
    if not query:
        return {
            "rewritten_query": "",
            "route_decision": "empty",
        }

    rewritten = query
    route = "rag"

    if len(query) < 8 or _VAGUE_PATTERN.search(query):
        try:
            model = get_light_chat_model()
            prompt = (
                "将用户问题改写为适合知识库检索的独立完整问句。"
                "只输出改写后的问题，不要解释。\n"
                f"用户问题：{query}"
            )
            resp = model.invoke([HumanMessage(content=prompt)])
            candidate = (resp.content or "").strip()
            if candidate and len(candidate) >= 2:
                rewritten = candidate
                route = "rag_rewritten"
        except Exception as exc:
            logger.warning(f"[Router] 查询改写失败，使用原问题: {exc}")
            route = "rag"

    return {
        "rewritten_query": rewritten,
        "route_decision": route,
    }
