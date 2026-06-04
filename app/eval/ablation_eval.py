"""
RAG 检索消融实验（Ablation Study）

对比不同检索/精排配置在固定数据集上的 Hit@K、MRR 等指标。

用法:
    python -m app.eval.ablation_eval
    python -m app.eval.ablation_eval --k 3 --dataset eval/retrieval_dataset.json
    python -m app.eval.ablation_eval --import-data
    python -m app.eval.ablation_eval --output eval/ablation_results.json
    python -m app.eval.ablation_eval --only A1_hybrid_legacy,A3_hybrid_rerank
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core import config_data as config
from app.core.logger import logger
from app.core.vector_stores import VectorStoreService
from app.eval.retrieval_eval import (
    CaseResult,
    evaluate_case,
    load_dataset,
    print_report,
    warn_dataset_coverage,
)
from app.eval.seed_data import import_data_directory, DEFAULT_DATA_DIR
from app.retrieval.pipeline import RetrievalPipelineConfig, retrieve_documents

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY


@dataclass
class AblationExperiment:
    id: str
    name: str
    description: str
    pipeline: RetrievalPipelineConfig


@dataclass
class AblationRunResult:
    experiment_id: str
    experiment_name: str
    description: str
    top_k: int
    hit_at_k_any: float
    hit_at_k_all: float
    mrr: float
    avg_keyword_recall: float
    source_hit_at_k: float | None
    delta_hit_any_vs_baseline: float | None = None
    case_results: list[CaseResult] = field(default_factory=list)


@dataclass
class AblationReport:
    dataset: str
    evaluated_at: str
    top_k: int
    baseline_id: str
    experiments: list[AblationRunResult]


def default_experiments() -> list[AblationExperiment]:
    k = config.SIMILARITY_THRESHOLD
    recall = config.RETRIEVAL_RECALL_K
    return [
        AblationExperiment(
            id="A0_dense",
            name="仅向量检索",
            description=f"Milvus 稠密 Top-{k}，无 BM25/RRF/Rerank",
            pipeline=RetrievalPipelineConfig(
                mode="dense", recall_k=k, top_k=k, use_rerank=False
            ),
        ),
        AblationExperiment(
            id="A1_hybrid_legacy",
            name="混合检索(旧版)",
            description="RRF + 97% 阈值过滤，最多 3 条（原玩具 RAG 策略）",
            pipeline=RetrievalPipelineConfig(
                mode="hybrid_legacy",
                recall_k=k,
                top_k=k,
                use_rerank=False,
                legacy_rrf_filter=True,
            ),
        ),
        AblationExperiment(
            id="A2_hybrid_wide",
            name="混合宽召回(已禁用生产)",
            description=f"RRF 宽召回 Top-{recall} 直接截 Top-{k}，无 Rerank（仅消融对比）",
            pipeline=RetrievalPipelineConfig(
                mode="hybrid",
                recall_k=recall,
                top_k=k,
                use_rerank=False,
                allow_wide_without_rerank=True,
            ),
        ),
        AblationExperiment(
            id="A3_hybrid_rerank",
            name="混合 + Rerank(生产默认)",
            description=f"RRF Top-{recall} → {config.RERANK_MODEL} 精排 Top-{k}（= production_pipeline）",
            pipeline=RetrievalPipelineConfig(
                mode="hybrid",
                recall_k=recall,
                top_k=k,
                use_rerank=True,
            ),
        ),
        AblationExperiment(
            id="A4_dense_rerank",
            name="向量 + Rerank",
            description=f"Milvus Top-{recall} → Rerank Top-{k}",
            pipeline=RetrievalPipelineConfig(
                mode="dense",
                recall_k=recall,
                top_k=k,
                use_rerank=True,
            ),
        ),
    ]


def run_single_experiment(
    exp: AblationExperiment,
    cases: list[dict],
    vector_service: VectorStoreService,
    top_k: int,
) -> AblationRunResult:
    case_results: list[CaseResult] = []
    for case in cases:
        docs = retrieve_documents(
            case["query"],
            vector_service,
            pipeline=exp.pipeline,
        )
        case_results.append(evaluate_case(docs, case, top_k))

    source_cases = [r for r in case_results if r.source_hit is not None]
    return AblationRunResult(
        experiment_id=exp.id,
        experiment_name=exp.name,
        description=exp.description,
        top_k=top_k,
        hit_at_k_any=sum(r.hit_any for r in case_results) / max(len(case_results), 1),
        hit_at_k_all=sum(r.hit_all for r in case_results) / max(len(case_results), 1),
        mrr=sum(r.mrr for r in case_results) / max(len(case_results), 1),
        avg_keyword_recall=sum(r.keyword_recall for r in case_results)
        / max(len(case_results), 1),
        source_hit_at_k=(
            sum(1 for r in source_cases if r.source_hit) / len(source_cases)
            if source_cases
            else None
        ),
        case_results=case_results,
    )


def run_ablation(
    dataset_path: Path,
    top_k: int,
    *,
    only_ids: list[str] | None = None,
    baseline_id: str = "A1_hybrid_legacy",
) -> AblationReport:
    cases = load_dataset(dataset_path)
    embeddings = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)
    try:
        vector_service = VectorStoreService(embeddings)
    except Exception as exc:
        raise RuntimeError(
            "无法连接 Milvus。请先停止 API 服务 (python -m app.api.api_service)，再运行消融实验。"
        ) from exc

    experiments = default_experiments()
    if only_ids:
        experiments = [e for e in experiments if e.id in only_ids]
        if not experiments:
            raise ValueError(f"未找到实验: {only_ids}")

    results: list[AblationRunResult] = []
    baseline_hit: float | None = None

    for exp in experiments:
        logger.info(f"[Ablation] 运行 {exp.id}: {exp.name}")
        run = run_single_experiment(exp, cases, vector_service, top_k)
        if exp.id == baseline_id:
            baseline_hit = run.hit_at_k_any
        results.append(run)

    if baseline_hit is not None:
        for run in results:
            run.delta_hit_any_vs_baseline = run.hit_at_k_any - baseline_hit

    return AblationReport(
        dataset=str(dataset_path),
        evaluated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        top_k=top_k,
        baseline_id=baseline_id,
        experiments=results,
    )


def print_ablation_report(report: AblationReport) -> None:
    print("\n" + "=" * 72)
    print("Mem-RAG 检索消融实验报告 (Ablation Study)")
    print("=" * 72)
    print(f"数据集: {report.dataset}")
    print(f"评估时间: {report.evaluated_at}")
    print(f"Top-K 评测: {report.top_k}")
    print(f"基线实验: {report.baseline_id}")
    print("-" * 72)
    print(
        f"{'ID':<20} {'Hit@K(任一)':>12} {'MRR':>8} {'关键词召回':>10} {'Δ基线':>8} 说明"
    )
    print("-" * 72)
    for run in report.experiments:
        delta = (
            f"{run.delta_hit_any_vs_baseline:+.1%}"
            if run.delta_hit_any_vs_baseline is not None
            else "  baseline"
        )
        print(
            f"{run.experiment_id:<20} "
            f"{run.hit_at_k_any:>11.1%} "
            f"{run.mrr:>8.3f} "
            f"{run.avg_keyword_recall:>10.1%} "
            f"{delta:>8} "
            f"{run.experiment_name}"
        )
    print("-" * 72)
    best = max(report.experiments, key=lambda r: r.hit_at_k_any)
    print(f"最佳 Hit@K(任一): {best.experiment_id} ({best.hit_at_k_any:.1%})")
    print("=" * 72 + "\n")


def save_ablation_report(report: AblationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    logger.info(f"[Ablation] 报告已保存: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mem-RAG 检索消融实验")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "eval" / "retrieval_dataset.json"),
    )
    parser.add_argument("--k", type=int, default=config.SIMILARITY_THRESHOLD)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "eval" / "ablation_results.json"))
    parser.add_argument("--baseline", default="A1_hybrid_legacy")
    parser.add_argument(
        "--only",
        default="",
        help="逗号分隔实验 ID，如 A2_hybrid_wide,A3_hybrid_rerank",
    )
    parser.add_argument("--import-data", action="store_true")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--detail", action="store_true", help="打印每个实验的逐条明细")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.import_data:
        print(f"正在导入: {args.data_dir}")
        for item in import_data_directory(Path(args.data_dir)):
            print(f"  {item['filename']}: {item['message']}")
        print()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    cases = load_dataset(dataset_path)
    warn_dataset_coverage(cases)

    only_ids = [x.strip() for x in args.only.split(",") if x.strip()] or None
    report = run_ablation(
        dataset_path,
        args.k,
        only_ids=only_ids,
        baseline_id=args.baseline,
    )
    print_ablation_report(report)

    if args.detail:
        for run in report.experiments:
            print(f"\n>>> {run.experiment_id}: {run.experiment_name}")
            from app.eval.retrieval_eval import EvalReport

            mini = EvalReport(
                dataset=report.dataset,
                evaluated_at=report.evaluated_at,
                top_k=run.top_k,
                total_cases=len(run.case_results),
                hit_at_k_any=run.hit_at_k_any,
                hit_at_k_all=run.hit_at_k_all,
                mrr=run.mrr,
                avg_keyword_recall=run.avg_keyword_recall,
                source_hit_at_k=run.source_hit_at_k,
                avg_retrieved_count=0,
                case_results=run.case_results,
            )
            print_report(mini)

    if args.output:
        save_ablation_report(report, Path(args.output))


if __name__ == "__main__":
    main()
