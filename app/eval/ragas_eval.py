from __future__ import annotations

import logging
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
            from ragas.metrics import (
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
