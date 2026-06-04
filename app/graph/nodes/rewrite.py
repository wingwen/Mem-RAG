"""Query Rewrite Agent：语义补全与意图重写，提升检索召回。"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core import config_data as config
from app.core.logger import logger
from app.core.structured_memory import get_structured_memory
from app.graph.state import AgentState
from app.llm.factory import get_light_chat_model

_PRONOUN_HINT = re.compile(r"(它|他|她|这个|那个|继续|还有呢|然后呢|刚才|上面)", re.I)


def _build_dialogue_snippet(session_id: str, query: str, max_turns: int = 3) -> str:
    """Rewrite 在 Memory 节点之前执行，此处轻量拉取近期对话与当前主题。"""
    mem = get_structured_memory()
    ctx = mem.build_context(session_id, query)
    lines = [f"当前讨论主题：{ctx.active_topic}"]
    for msg in (ctx.history_messages or [])[-max_turns * 2 :]:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if isinstance(msg, BaseMessage) else str(msg)
        text = (content or "")[:200]
        lines.append(f"{role}：{text}")
    return "\n".join(lines) if len(lines) > 1 else "（暂无历史对话）"


def _needs_rewrite(query: str) -> bool:
    q = query.strip()
    if len(q) < 12:
        return True
    if _PRONOUN_HINT.search(q):
        return True
    return False


def rewrite_node(state: AgentState) -> dict:
    """
    将省略主语、指代、过短问句改写为可独立检索的完整问句。

    例：「它支持 Docker 吗」→「Mem-RAG 是否支持 Docker 部署」
    """
    query = (state.get("query") or "").strip()
    session_id = state.get("session_id") or ""

    if not query:
        return {
            "rewritten_query": "",
            "route_decision": "empty",
            "retrieval_status": ["[状态] Query Rewrite：空问题，跳过改写\n"],
        }

    rewritten = query
    route = "rewrite_skip"

    if _needs_rewrite(query):
        try:
            dialogue = _build_dialogue_snippet(session_id, query)
            model = get_light_chat_model()
            prompt = f"""你是 Query Rewrite Agent，负责把用户问题改写成适合知识库检索的**独立完整问句**。

规则：
1. 补全主语与实体（如「它」→ 结合对话推断的具体产品/主题，默认产品名：{config.REWRITE_PROJECT_NAME}）
2. 明确用户意图（部署、功能、原理、对比等）
3. 保留关键检索词（专有名词、技术术语如 Docker、Milvus）
4. 只输出一行改写后的问题，不要解释、不要 JSON

近期对话：
{dialogue}

用户原始问题：{query}

改写后的问题："""
            resp = model.invoke([HumanMessage(content=prompt)])
            candidate = (resp.content or "").strip()
            candidate = candidate.strip('"\'「」')
            if candidate.startswith("{"):
                try:
                    data = json.loads(candidate)
                    candidate = data.get("rewritten_query") or data.get("query") or candidate
                except json.JSONDecodeError:
                    pass
            if candidate and len(candidate) >= 2 and candidate != query:
                rewritten = candidate
                route = "rewrite_applied"
            else:
                route = "rewrite_unchanged"
        except Exception as exc:
            logger.warning(f"[Rewrite] 查询改写失败，使用原问题: {exc}")
            route = "rewrite_fallback"
    else:
        route = "rewrite_skip"

    status = (
        f"[状态] Query Rewrite：{query} → {rewritten}\n"
        if rewritten != query
        else f"[状态] Query Rewrite：保持原问句\n"
    )

    return {
        "rewritten_query": rewritten,
        "route_decision": route,
        "retrieval_status": [status],
    }
