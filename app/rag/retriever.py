from __future__ import annotations

import logging

from langchain_core.documents import Document

from app.core.config import Settings
from app.core.errors import RetrievalError
from app.rag.bm25_store import BM25Store
from app.rag.reranker import RerankerService
from app.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RetrieverService:
    def __init__(
        self,
        settings: Settings,
        vector_store_manager: VectorStoreManager,
        bm25_store: BM25Store,
        reranker: RerankerService | None = None,
    ):
        self._settings = settings
        self._vector_store_manager = vector_store_manager
        self._bm25_store = bm25_store
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        search_type: str = "similarity",
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        reranker_enabled: bool = False,
    ) -> list[tuple[Document, float | None]]:
        try:
            k = top_k or self._settings.search_top_k
            fetch_k = max(k * 3, 20) if reranker_enabled else k

            if search_type == "bm25":
                results = self._bm25_search(query, fetch_k)
            elif search_type == "hybrid":
                results = self._hybrid_search(
                    query, fetch_k, vector_weight, bm25_weight
                )
            elif search_type == "mmr":
                results = self._mmr_search(query, fetch_k)
            else:
                results = self._vector_search(query, fetch_k)

            if score_threshold is not None and search_type not in ("mmr",):
                results = [
                    (doc, score) for doc, score in results
                    if score is None or score >= score_threshold
                ]

            if reranker_enabled and self._reranker is not None:
                results = self._reranker.rerank(query, results, top_k=k)
            else:
                results = results[:k]

            return results
        except Exception as exc:
            raise RetrievalError(f"检索失败: {exc}") from exc

    def _vector_search(
        self, query: str, k: int
    ) -> list[tuple[Document, float | None]]:
        pairs = self._vector_store_manager.store.similarity_search_with_relevance_scores(
            query, k=k
        )
        return [(doc, score) for doc, score in pairs]

    def _mmr_search(
        self, query: str, k: int
    ) -> list[tuple[Document, float | None]]:
        docs = self._vector_store_manager.store.max_marginal_relevance_search(
            query, k=k, fetch_k=max(k * 4, 20)
        )
        return [(doc, None) for doc in docs]

    def _bm25_search(
        self, query: str, k: int
    ) -> list[tuple[Document, float | None]]:
        return self._bm25_store.search(query, top_k=k)

    def _hybrid_search(
        self,
        query: str,
        k: int,
        vector_weight: float,
        bm25_weight: float,
    ) -> list[tuple[Document, float | None]]:
        vector_results = self._vector_search(query, k)
        bm25_results = self._bm25_search(query, k)

        vector_scores = _normalize_scores(vector_results)
        bm25_scores = _normalize_scores(bm25_results)

        combined: dict[str, tuple[Document, float]] = {}

        for doc, norm_score in vector_scores:
            key = _doc_key(doc)
            combined[key] = (doc, vector_weight * norm_score)

        for doc, norm_score in bm25_scores:
            key = _doc_key(doc)
            if key in combined:
                existing_doc, existing_score = combined[key]
                combined[key] = (existing_doc, existing_score + bm25_weight * norm_score)
            else:
                combined[key] = (doc, bm25_weight * norm_score)

        ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return ranked[:k]


def _doc_key(doc: Document) -> str:
    chunk_id = doc.metadata.get("chunk_id")
    source = doc.metadata.get("source", "")
    if chunk_id is not None:
        return f"{source}::{chunk_id}"
    return doc.page_content[:200]


def _normalize_scores(
    pairs: list[tuple[Document, float | None]],
) -> list[tuple[Document, float]]:
    """Min-max normalize scores to [0, 1]."""
    scores = [s for _, s in pairs if s is not None]
    if not scores:
        return [(doc, 0.0) for doc, _ in pairs]

    min_s, max_s = min(scores), max(scores)
    span = max_s - min_s

    if span == 0:
        return [(doc, 1.0) for doc, _ in pairs]

    return [
        (doc, (s - min_s) / span if s is not None else 0.0)
        for doc, s in pairs
    ]
