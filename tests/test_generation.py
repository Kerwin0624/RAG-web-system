from __future__ import annotations

from app.rag.pipeline import RAGPipeline


class DummyRetriever:
    def retrieve(self, query, top_k, score_threshold, search_type,
                 vector_weight=0.7, bm25_weight=0.3, reranker_enabled=False):
        class Doc:
            page_content = "这是检索到的上下文内容。"
            metadata = {"source": "unit-test"}

        return [(Doc(), 0.95)]


class DummyGenerator:
    def generate(self, question, contexts, temperature, max_tokens, top_p, model=None):
        return f"Q={question};C={contexts[0]}"


def test_pipeline_cache():
    pipeline = RAGPipeline(retriever=DummyRetriever(), generator=DummyGenerator(), cache_size=2)
    first = pipeline.ask("什么是RAG", 4, 0.2, "similarity", temperature=0.1, max_tokens=256, top_p=1.0)
    second = pipeline.ask("什么是RAG", 4, 0.2, "similarity", temperature=0.1, max_tokens=256, top_p=1.0)

    assert first["answer"]
    assert second["from_cache"] is True
