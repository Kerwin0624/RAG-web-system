from __future__ import annotations

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RerankerService:
    """Cross-encoder reranker with lazy model loading."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model_name = model_name
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for reranking. "
                "Install with: pip install 'rag-web-system[reranker]'"
            ) from exc
        logger.info("Loading reranker model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        logger.info("Reranker model loaded")

    def rerank(
        self,
        query: str,
        docs_with_scores: list[tuple[Document, float | None]],
        top_k: int = 4,
    ) -> list[tuple[Document, float]]:
        if not docs_with_scores:
            return []

        self._ensure_model()

        pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
        rerank_scores = self._model.predict(pairs)

        reranked = [
            (doc, float(score))
            for (doc, _), score in zip(docs_with_scores, rerank_scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
