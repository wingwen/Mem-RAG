"""
检索评估脚本：Hit@K、MRR、关键词召回率、来源命中率。

用法:
    python -m app.eval.retrieval_eval
    python -m app.eval.retrieval_eval --import-data
    python -m app.eval.retrieval_eval --k 5 --dataset eval/retrieval_dataset.json
    python -m app.eval.retrieval_eval --dataset eval/nfs_dataset.json
    python -m app.eval.retrieval_eval --output eval/results.json

    python -m app.eval.seed_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.embeddings import DashScopeEmbeddings

from app.core import config_data as config
from app.core.vector_stores import VectorStoreService
from app.core.logger import logger
from app.eval.seed_data import import_data_directory, DEFAULT_DATA_DIR

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY

MatchMode = Literal["any", "all"]


@dataclass
class CaseResult:
    case_id: str
    query: str
    retrieved_count: int
    hit_any: bool
    hit_all: bool
    source_hit: bool | None
    mrr: float
    keyword_recall: float
    top_sources: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    dataset: str
    evaluated_at: str
    top_k: int
    total_cases: int
    hit_at_k_any: float
    hit_at_k_all: float
    mrr: float
    avg_keyword_recall: float
    source_hit_at_k: float | None
    avg_retrieved_count: float
    case_results: list[CaseResult]


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["cases"]


def _keywords_in_text(keywords: list[str], text: str, mode: MatchMode) -> bool:
    if not keywords:
        return True
    if mode == "all":
        return all(kw in text for kw in keywords)
    return any(kw in text for kw in keywords)


def _collect_keyword_stats(keywords: list[str], docs: list[Document], k: int) -> tuple[list[str], list[str], float]:
    if not keywords:
        return [], [], 1.0
    merged = "\n".join(doc.page_content for doc in docs[:k])
    matched = [kw for kw in keywords if kw in merged]
    missing = [kw for kw in keywords if kw not in merged]
    recall = len(matched) / len(keywords)
    return matched, missing, recall


def evaluate_case(docs: list[Document], case: dict, k: int) -> CaseResult:
    keywords = case.get("expected_keywords", [])
    expected_source = case.get("expected_source")
    top_docs = docs[:k]

    hit_any = any(_keywords_in_text(keywords, doc.page_content, "any") for doc in top_docs)
    hit_all = _keywords_in_text(keywords, "\n".join(doc.page_content for doc in top_docs), "all")

    mrr = 0.0
    for rank, doc in enumerate(top_docs, 1):
        if _keywords_in_text(keywords, doc.page_content, "any"):
            mrr = 1.0 / rank
            break

    source_hit = None
    if expected_source:
        source_hit = any(doc.metadata.get("filename") == expected_source for doc in top_docs)

    matched, missing, keyword_recall = _collect_keyword_stats(keywords, docs, k)
    top_sources = [doc.metadata.get("filename", "unknown") for doc in top_docs]

    return CaseResult(
        case_id=case["id"],
        query=case["query"],
        retrieved_count=len(docs),
        hit_any=hit_any,
        hit_all=hit_all,
        source_hit=source_hit,
        mrr=mrr,
        keyword_recall=keyword_recall,
        top_sources=top_sources,
        matched_keywords=matched,
        missing_keywords=missing,
    )


def run_evaluation(dataset_path: Path, top_k: int) -> EvalReport:
    cases = load_dataset(dataset_path)
    embeddings = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)
    try:
        vector_service = VectorStoreService(embeddings)
    except Exception as exc:
        raise RuntimeError(
            "无法连接 Milvus 向量库。请先停止正在运行的 API 服务 "
            "(python -m app.api.api_service)，再重新执行评估。"
        ) from exc
    retriever = vector_service.get_retriever()

    case_results: list[CaseResult] = []
    for case in cases:
        docs = retriever.invoke(case["query"])
        case_results.append(evaluate_case(docs, case, top_k))

    source_cases = [r for r in case_results if r.source_hit is not None]
    report = EvalReport(
        dataset=str(dataset_path),
        evaluated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        top_k=top_k,
        total_cases=len(case_results),
        hit_at_k_any=sum(r.hit_any for r in case_results) / max(len(case_results), 1),
        hit_at_k_all=sum(r.hit_all for r in case_results) / max(len(case_results), 1),
        mrr=sum(r.mrr for r in case_results) / max(len(case_results), 1),
        avg_keyword_recall=sum(r.keyword_recall for r in case_results) / max(len(case_results), 1),
        source_hit_at_k=(
            sum(1 for r in source_cases if r.source_hit) / len(source_cases) if source_cases else None
        ),
        avg_retrieved_count=sum(r.retrieved_count for r in case_results) / max(len(case_results), 1),
        case_results=case_results,
    )
    return report


def print_report(report: EvalReport) -> None:
    print("\n" + "=" * 56)
    print("Mem-RAG 检索评估报告")
    print("=" * 56)
    print(f"数据集: {report.dataset}")
    print(f"评估时间: {report.evaluated_at}")
    print(f"Top-K: {report.top_k}")
    print(f"样本数: {report.total_cases}")
    print("-" * 56)
    print(f"Hit@{report.top_k} (任一关键词): {report.hit_at_k_any:.2%}")
    print(f"Hit@{report.top_k} (全部关键词): {report.hit_at_k_all:.2%}")
    print(f"MRR: {report.mrr:.4f}")
    print(f"平均关键词召回率: {report.avg_keyword_recall:.2%}")
    if report.source_hit_at_k is not None:
        print(f"来源 Hit@{report.top_k}: {report.source_hit_at_k:.2%}")
    print(f"平均返回文档数: {report.avg_retrieved_count:.2f}")
    print("-" * 56)
    print("逐条结果:")
    for item in report.case_results:
        status = "PASS" if item.hit_any else "FAIL"
        print(f"[{status}] {item.case_id} | {item.query}")
        print(f"       来源: {item.top_sources}")
        if item.matched_keywords:
            print(f"       命中关键词: {item.matched_keywords}")
        if item.missing_keywords:
            print(f"       未命中关键词: {item.missing_keywords}")
    print("=" * 56 + "\n")


def save_report(report: EvalReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[Eval] 报告已保存: {output_path}")


def list_kb_filenames() -> set[str]:
    from pymilvus import MilvusClient

    client = MilvusClient(config.MILVUS_URI)
    if not client.has_collection(config.COLLECTION_NAME):
        return set()
    client.load_collection(collection_name=config.COLLECTION_NAME)
    rows = client.query(
        collection_name=config.COLLECTION_NAME,
        filter="",
        output_fields=["filename"],
        limit=1000,
    )
    return {row.get("filename", "") for row in rows if row.get("filename")}


def warn_dataset_coverage(cases: list[dict]) -> None:
    kb_files = list_kb_filenames()
    if not kb_files:
        logger.warning("[Eval] 知识库为空，请先上传文档或执行 --import-data")
        return

    expected = {case["expected_source"] for case in cases if case.get("expected_source")}
    missing = sorted(expected - kb_files)
    if missing:
        print("\n⚠ 警告：以下期望来源不在当前知识库中：")
        for name in missing:
            print(f"   - {name}")
        print(f"   当前知识库文件: {sorted(kb_files)}")
        print("   可先运行: python -m app.eval.retrieval_eval --import-data\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mem-RAG 检索评估")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "eval" / "retrieval_dataset.json"),
        help="评估数据集 JSON 路径",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=config.SIMILARITY_THRESHOLD,
        help="Top-K 检索数量",
    )
    parser.add_argument(
        "--output",
        default="",
        help="可选，保存 JSON 报告的路径",
    )
    parser.add_argument(
        "--import-data",
        action="store_true",
        help="评估前先将 data/ 目录下的 txt 导入知识库",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="--import-data 使用的数据目录，默认 data/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.import_data:
        print(f"正在导入数据目录: {args.data_dir}")
        results = import_data_directory(Path(args.data_dir))
        for item in results:
            print(f"  {item['filename']}: {item['message']}")
        print()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    cases = load_dataset(dataset_path)
    warn_dataset_coverage(cases)

    report = run_evaluation(dataset_path, args.k)
    print_report(report)

    if args.output:
        save_report(report, Path(args.output))


if __name__ == "__main__":
    main()
