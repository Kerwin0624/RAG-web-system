from __future__ import annotations

import time
from collections import OrderedDict
from hashlib import md5
from threading import Lock

from app.rag.generator import LLMGenerator
from app.rag.retriever import RetrieverService


class RAGPipeline:
    def __init__(self, retriever: RetrieverService, generator: LLMGenerator, cache_size: int = 256):
        self._retriever = retriever
        self._generator = generator
        self._cache_size = cache_size
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_lock = Lock()

    def _cache_key(self, **kwargs: object) -> str:
        payload = "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return md5(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict | None:
        with self._cache_lock:
            if key not in self._cache:
                return None
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

    def _write_cache(self, key: str, value: dict) -> None:
        with self._cache_lock:
            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = value
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def ask(
        self,
        question: str,
        top_k: int,
        score_threshold: float | None,
        search_type: str,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        reranker_enabled: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,
        model: str | None = None,
    ) -> dict:
        started = time.perf_counter()
        key = self._cache_key(
            question=question,
            top_k=top_k,
            score_threshold=score_threshold,
            search_type=search_type,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            reranker_enabled=reranker_enabled,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model or "",
        )
        cached = self._read_cache(key)
        if cached is not None:
            result = dict(cached)
            result["from_cache"] = True
            return result

        retrieved = self._retriever.retrieve(
            query=question,
            top_k=top_k,
            score_threshold=score_threshold,
            search_type=search_type,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            reranker_enabled=reranker_enabled,
        )
        contexts = [doc.page_content for doc, _ in retrieved]
        answer = self._generator.generate(
            question=question,
            contexts=contexts,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        citations = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "score": score,
            }
            for doc, score in retrieved
        ]
        result = {
            "answer": answer,
            "citations": citations,
            "elapsed_ms": elapsed_ms,
            "from_cache": False,
        }
        self._write_cache(key, result)
        return result
