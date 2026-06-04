"""结构化记忆服务（封装 StructuredMemory）。"""

from __future__ import annotations

from typing import Any

from app.core.structured_memory import MemoryContext, get_structured_memory


class MemoryService:
    def recall(self, session_id: str, query: str) -> MemoryContext:
        return get_structured_memory().build_context(session_id, query)

    def to_state_payload(self, ctx: MemoryContext) -> dict[str, Any]:
        return {
            "memories": [
                {
                    "active_topic": ctx.active_topic,
                    "active_topic_id": ctx.active_topic_id,
                }
            ],
            "memory_block": ctx.memory_block,
            "history_messages": ctx.history_messages,
            "active_topic": ctx.active_topic,
        }
