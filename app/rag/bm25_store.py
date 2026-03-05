from __future__ import annotations

import logging
import pickle
from pathlib import Path

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Store:
    """BM25 keyword index with jieba tokenization and pickle persistence."""

    def __init__(self, corpus_path: Path):
        self._corpus_path = corpus_path
        self._documents: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    @property
    def size(self) -> int:
        return len(self._documents)

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        for doc in docs:
            tokens = self._tokenize(doc.page_content)
            self._documents.append(doc)
            self._tokenized_corpus.append(tokens)
        self._rebuild_index()
        self._save()

    def search(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        if self._bm25 is None or not self._documents:
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        scored_pairs = [(self._documents[i], float(scores[i])) for i in range(len(scores))]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        return [w for w in jieba.cut(text) if w.strip()]

    def _rebuild_index(self) -> None:
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def _save(self) -> None:
        try:
            self._corpus_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "documents": [(d.page_content, d.metadata) for d in self._documents],
                "tokenized_corpus": self._tokenized_corpus,
            }
            with open(self._corpus_path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            logger.warning("Failed to persist BM25 corpus", exc_info=True)

    def _load(self) -> None:
        if not self._corpus_path.exists():
            return
        try:
            with open(self._corpus_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            self._documents = [
                Document(page_content=content, metadata=meta)
                for content, meta in data["documents"]
            ]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._rebuild_index()
            logger.info("Loaded BM25 corpus with %d documents", len(self._documents))
        except Exception:
            logger.warning("Failed to load BM25 corpus, starting fresh", exc_info=True)
            self._documents = []
            self._tokenized_corpus = []
            self._bm25 = None
