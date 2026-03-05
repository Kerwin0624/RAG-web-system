from __future__ import annotations

import logging
from typing import Any

from app.core.errors import EvaluationError
from app.eval.dataset_builder import EvalDatasetBuilder
from app.schemas.rag import EvalSample

logger = logging.getLogger(__name__)


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

    def evaluate_samples(self, samples: list[EvalSample]) -> dict[str, float | str]:
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
        except Exception as exc:  # pragma: no cover - environment dependent
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
            raw = result.to_pandas().mean(numeric_only=True).to_dict()
            return {k: float(v) for k, v in raw.items()}
        except Exception as exc:
            raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc
