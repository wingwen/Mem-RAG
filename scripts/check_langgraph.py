"""LangGraph 构建自检脚本。用法: python scripts/check_langgraph.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    errors = []

    print("=" * 50)
    print("Mem-RAG LangGraph 构建检查")
    print("=" * 50)

    # 1. 模块导入
    try:
        from app.graph.state import AgentState
        from app.graph.workflow import build_agent_graph, get_agent_graph
        from app.graph.nodes import (
            router_node,
            memory_node,
            retrieval_node,
            fusion_node,
            generation_node,
        )
        from app.memory.service import MemoryService
        from app.retrieval.service import RetrievalService
        from app.llm.factory import get_chat_model, get_light_chat_model
        print("[OK] 模块导入")
    except Exception as e:
        errors.append(f"模块导入失败: {e}")
        print(f"[FAIL] 模块导入: {e}")

    # 2. 图编译
    try:
        g_prep = get_agent_graph(include_generation=False)
        g_full = build_agent_graph(include_generation=True)
        nodes_prep = [n for n in g_prep.get_graph().nodes.keys() if not n.startswith("__")]
        nodes_full = [n for n in g_full.get_graph().nodes.keys() if not n.startswith("__")]
        from app.core import config_data as cfg

        if cfg.RERANK_ENABLED:
            expected_prep = ["router", "memory", "retrieval", "rerank", "fusion"]
        else:
            expected_prep = ["router", "memory", "retrieval", "fusion"]
        expected_full = expected_prep + ["generation"]
        if nodes_prep != expected_prep:
            errors.append(f"Prep 节点不符: {nodes_prep} != {expected_prep}")
            print(f"[FAIL] Prep 图节点: {nodes_prep}")
        else:
            print(f"[OK] Prep 图节点: {nodes_prep}")
        if nodes_full != expected_full:
            errors.append(f"Full 节点不符: {nodes_full} != {expected_full}")
            print(f"[FAIL] Full 图节点: {nodes_full}")
        else:
            print(f"[OK] Full 图节点: {nodes_full}")
    except Exception as e:
        errors.append(f"图编译失败: {e}")
        print(f"[FAIL] 图编译: {e}")

    # 3. RagService 初始化（会连 Milvus，需 database/ 可访问）
    try:
        from app.core.rag import RagService

        svc = RagService()
        print("[OK] RagService 初始化")
    except Exception as e:
        errors.append(f"RagService 初始化失败: {e}")
        print(f"[FAIL] RagService: {e}")
        print("      提示: 确认 .env 中 DASHSCOPE_API_KEY 已配置")

    # 4. AgentState 字段
    try:
        hints = AgentState.__annotations__
        required = [
            "query",
            "rewritten_query",
            "memories",
            "retrieved_docs",
            "final_context",
            "answer",
        ]
        missing = [k for k in required if k not in hints]
        if missing:
            errors.append(f"AgentState 缺少字段: {missing}")
            print(f"[FAIL] AgentState 字段缺失: {missing}")
        else:
            print(f"[OK] AgentState 核心字段齐全")
    except Exception as e:
        errors.append(f"AgentState 检查失败: {e}")

    print("=" * 50)
    if errors:
        print("结果: 未通过")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("结果: 全部通过（未测端到端对话，需启动服务后手动验证）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
