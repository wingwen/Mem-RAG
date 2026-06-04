"""Memory Recall：结构化主题记忆召回。"""

from app.graph.state import AgentState
from app.memory.service import MemoryService


def memory_node(state: AgentState) -> dict:
    session_id = state.get("session_id") or ""
    # Memory 在 Rewrite 之后：召回结构化记忆时参考改写后的检索意图
    query = state.get("rewritten_query") or state.get("query") or ""

    svc = MemoryService()
    ctx = svc.recall(session_id, query)
    payload = svc.to_state_payload(ctx)

    return {
        **payload,
        "retrieval_status": ["[状态] 结构化记忆已加载\n"],
    }
