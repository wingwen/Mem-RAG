"""Context Fusion：记忆块 + 检索文档 → final_context。"""

from app.graph.state import AgentState
from app.retrieval.service import RetrievalService


def fusion_node(state: AgentState) -> dict:
    memory_block = state.get("memory_block") or ""
    docs = state.get("retrieved_docs") or []
    ref_text = RetrievalService.format_documents(docs)

    final_context = (
        f"{memory_block}\n\n"
        f"【检索参考资料】\n{ref_text}"
    ).strip()

    return {
        "final_context": final_context,
        "retrieval_status": ["[状态] 上下文融合完成\n"],
    }
