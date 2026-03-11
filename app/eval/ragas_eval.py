from __future__ import annotations

import logging
import os
from typing import Any

from app.core.errors import EvaluationError
from app.eval.dataset_builder import EvalDatasetBuilder
from app.schemas.rag import EvalSample

logger = logging.getLogger(__name__)


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    if total == 0:
        return 0.0
    return 2.0 * precision * recall / total


# 默认指标目标（得分低于目标时给出优化建议）
DEFAULT_TARGET_FAITHFULNESS = 0.85
DEFAULT_TARGET_ANSWER_RELEVANCY = 0.8
DEFAULT_TARGET_CONTEXT_RECALL = 0.8
DEFAULT_TARGET_CONTEXT_PRECISION = 0.7


def get_optimization_suggestions(
    summary: dict[str, float],
    *,
    target_faithfulness: float = DEFAULT_TARGET_FAITHFULNESS,
    target_answer_relevancy: float = DEFAULT_TARGET_ANSWER_RELEVANCY,
    target_context_recall: float = DEFAULT_TARGET_CONTEXT_RECALL,
    target_context_precision: float = DEFAULT_TARGET_CONTEXT_PRECISION,
) -> list[str]:
    """根据 RAGAS 汇总指标与用户设定的目标，对未达标项生成优化方向建议。"""
    suggestions: list[str] = []
    cp = summary.get("context_precision", 1.0)
    cr = summary.get("context_recall", 1.0)
    fa = summary.get("faithfulness", 1.0)
    ar = summary.get("answer_relevancy", 1.0)
    rf1 = summary.get("retrieval_f1", 1.0)

    if cp < target_context_precision:
        suggestions.append(
            f"**Context Precision 未达目标（当前 {cp:.3f} < {target_context_precision}）**："
            "检索到的片段中无关内容较多。建议：启用重排序（Reranker）、适当提高相似度阈值，或优化分块/分隔符减少噪声。"
        )
    if cr < target_context_recall:
        suggestions.append(
            f"**Context Recall 未达目标（当前 {cr:.3f} < {target_context_recall}）**："
            "相关文档未充分检索到。建议：适当增大 Top-K、尝试混合检索或 MMR，或调整分块大小/重叠以更好覆盖知识点。"
        )
    if fa < target_faithfulness:
        suggestions.append(
            f"**Faithfulness 未达目标（当前 {fa:.3f} < {target_faithfulness}）**："
            "回答存在未基于上下文的内容。建议：强化「仅依据上下文作答」的 prompt、适当降低温度，或先提升检索质量再生成。"
        )
    if ar < target_answer_relevancy:
        suggestions.append(
            f"**Answer Relevancy 未达目标（当前 {ar:.3f} < {target_answer_relevancy}）**："
            "回答与问题匹配度不足。建议：检查问题表述与知识库覆盖范围、优化 prompt 或尝试更强模型。"
        )
    rf1_target = min(target_context_precision, target_context_recall)
    if rf1 < rf1_target and not (cp < target_context_precision or cr < target_context_recall):
        suggestions.append(
            f"**检索综合表现未达预期（Retrieval F1 {rf1:.3f}）**："
            "可尝试启用重排序、混合检索，或微调 Top-K 与相似度阈值。"
        )
    if not suggestions:
        suggestions.append("当前 RAGAS 各维度均达到或超过设定目标，可维持现有参数或针对业务场景做小幅调优。")
    return suggestions


class RagasEvaluator:
    def __init__(self, llm: Any = None, embeddings: Any = None):
        self._llm = llm
        self._embeddings = embeddings

    @staticmethod
    def _wrap_llm(llm: Any) -> Any:
        """Wrap a LangChain LLM into a RAGAS-compatible object."""
        try:
            from ragas.llms import LangchainLLMWrapper
            return LangchainLLMWrapper(llm)
        except Exception:
            return llm

    @staticmethod
    def _wrap_embeddings(embeddings: Any) -> Any:
        """Wrap LangChain Embeddings into a RAGAS-compatible object."""
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            return LangchainEmbeddingsWrapper(embeddings)
        except Exception:
            return embeddings

    def evaluate_samples(self, samples: list[EvalSample]) -> dict[str, Any]:
        """Return both summary (mean) and per-sample RAGAS scores.

        Returns dict with keys:
          - ``summary``: mean scores + retrieval_f1
          - ``per_sample``: list of per-question score dicts
        """
        if not samples:
            raise EvaluationError("No evaluation samples provided.")

        try:
            from ragas import evaluate
            try:
                from ragas.metrics.collections import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )
            except Exception:
                from ragas.metrics import (  # pragma: no cover
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": f"RAGAS not available or import failed: {exc}",
            }

        if self._llm is None and not os.environ.get("OPENAI_API_KEY"):
            return {
                "status": "skipped",
                "reason": "未配置评估模型：请提供 evaluator llm 或设置 OPENAI_API_KEY。",
            }

        dataset = EvalDatasetBuilder.build(samples)
        try:
            extra: dict[str, Any] = {}
            if self._llm is not None:
                extra["llm"] = self._wrap_llm(self._llm)
                logger.info("RAGAS using custom LLM: %s", type(self._llm).__name__)
            if self._embeddings is not None:
                extra["embeddings"] = self._wrap_embeddings(self._embeddings)
                logger.info("RAGAS using custom embeddings: %s", type(self._embeddings).__name__)

            result = evaluate(
                dataset=dataset,
                metrics=[answer_relevancy, context_precision, context_recall, faithfulness],
                **extra,
            )

            df = result.to_pandas()

            per_sample: list[dict[str, float]] = []
            for _, row in df.iterrows():
                rec: dict[str, float] = {}
                for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
                    if col in row:
                        rec[col] = float(row[col]) if row[col] == row[col] else 0.0
                rec["retrieval_f1"] = _f1(
                    rec.get("context_precision", 0.0),
                    rec.get("context_recall", 0.0),
                )
                per_sample.append(rec)

            raw_mean = df.mean(numeric_only=True).to_dict()
            summary = {k: float(v) for k, v in raw_mean.items()}
            summary["retrieval_f1"] = _f1(
                summary.get("context_precision", 0.0),
                summary.get("context_recall", 0.0),
            )

            return {"summary": summary, "per_sample": per_sample}
        except Exception as exc:
            raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc
