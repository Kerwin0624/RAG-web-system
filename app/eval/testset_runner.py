from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.eval.ragas_eval import RagasEvaluator
from app.rag.pipeline import RAGPipeline
from app.schemas.rag import EvalSample

logger = logging.getLogger(__name__)


class TestsetRunner:
    """Load a JSON test set, run every question through the RAG pipeline,
    collect model answers / retrieved contexts, and optionally score with RAGAS."""

    def __init__(self, pipeline: RAGPipeline, evaluator: RagasEvaluator):
        self._pipeline = pipeline
        self._evaluator = evaluator

    def run(
        self,
        testset_path: str | Path,
        search_type: str = "similarity",
        top_k: int = 4,
        score_threshold: float = 0.25,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        reranker_enabled: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,
        run_ragas: bool = False,
    ) -> dict[str, Any]:
        path = Path(testset_path)
        with path.open("r", encoding="utf-8") as f:
            items: list[dict] = json.load(f)

        per_question: list[dict] = []
        eval_samples: list[EvalSample] = []
        total_start = time.perf_counter()

        for i, item in enumerate(items, 1):
            q = item["question"]
            gold = item["gold_answer"]
            logger.info("Evaluating [%d/%d]: %s", i, len(items), q[:60])

            try:
                result = self._pipeline.ask(
                    question=q,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    search_type=search_type,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                    reranker_enabled=reranker_enabled,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                model_answer = result["answer"]
                contexts = [c["content"] for c in result["citations"]]
                elapsed = result["elapsed_ms"]
                status = "success"
            except Exception as exc:
                logger.warning("Question failed: %s — %s", q[:60], exc)
                model_answer = f"[错误] {exc}"
                contexts = []
                elapsed = 0
                status = "error"

            per_question.append({
                "id": item.get("id", f"q-{i}"),
                "question": q,
                "gold_answer": gold,
                "model_answer": model_answer,
                "contexts_count": len(contexts),
                "elapsed_ms": elapsed,
                "question_type": item.get("question_type", ""),
                "difficulty": item.get("difficulty", ""),
                "has_answer": item.get("has_answer", True),
                "status": status,
            })

            if status == "success":
                eval_samples.append(EvalSample(
                    question=q,
                    answer=model_answer,
                    ground_truth=gold,
                    contexts=contexts if contexts else ["无检索结果"],
                ))

        total_elapsed_ms = int((time.perf_counter() - total_start) * 1000)

        ragas_metrics: dict[str, Any] = {}
        if run_ragas and eval_samples:
            try:
                ragas_metrics = self._evaluator.evaluate_samples(eval_samples)
            except Exception as exc:
                logger.exception("RAGAS evaluation failed")
                ragas_metrics = {"status": "error", "reason": str(exc)}

        success_count = sum(1 for r in per_question if r["status"] == "success")
        fail_count = sum(1 for r in per_question if r["status"] == "error")

        return {
            "total_questions": len(items),
            "successful": success_count,
            "failed": fail_count,
            "total_elapsed_ms": total_elapsed_ms,
            "per_question": per_question,
            "ragas_metrics": ragas_metrics,
        }
