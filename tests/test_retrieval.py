from pathlib import Path

from app.core.config import Settings
from app.rag.bm25_store import BM25Store
from app.rag.ingest import IngestService
from app.rag.retriever import RetrieverService


class DummyVectorStore:
    def __init__(self):
        self.docs = []

    def add_documents(self, docs):
        self.docs.extend(docs)

    def similarity_search_with_relevance_scores(self, query, k=4):
        scored = []
        for d in self.docs[:k]:
            score = 0.9 if query in d.page_content else 0.3
            scored.append((d, score))
        return scored

    def max_marginal_relevance_search(self, query, k=4, fetch_k=20):
        return self.docs[:k]


class DummyManager:
    def __init__(self):
        self.store = DummyVectorStore()

    def add_documents(self, docs):
        self.store.add_documents(docs)
        return len(docs)


def test_ingest_and_retrieve(tmp_path: Path):
    test_file = tmp_path / "doc.txt"
    test_file.write_text("LangChain 是一个用于构建LLM应用的框架。", encoding="utf-8")

    settings = Settings(
        LLM_BASE_URL="https://example.com/v1",
        LLM_API_KEY="x",
        LLM_MODEL="mock-model",
    )
    manager = DummyManager()
    bm25 = BM25Store(corpus_path=tmp_path / "bm25_corpus.pkl")
    ingest = IngestService(settings=settings, vector_store_manager=manager, bm25_store=bm25)
    count, _ = ingest.ingest(paths=[str(test_file)])
    assert count >= 1

    retriever = RetrieverService(settings=settings, vector_store_manager=manager, bm25_store=bm25)
    hits = retriever.retrieve("LangChain", top_k=2, score_threshold=0.5)
    assert len(hits) >= 1
