"""检索未命中：直接返回固定拒答，不调用 LLM。"""

from app.core import config_data as config
from app.graph.state import AgentState


def no_answer_node(state: AgentState) -> dict:
    message = config.NO_KB_ANSWER_MESSAGE
    return {
        "retrieval_hit": False,
        "retrieved_docs": [],
        "final_context": "",
        "answer": message,
        "retrieval_status": ["[状态] 检索未命中知识库，已跳过生成\n"],
    }
