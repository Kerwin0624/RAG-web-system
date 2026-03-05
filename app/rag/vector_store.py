from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.core.config import Settings


class VectorStoreManager:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._embeddings = OllamaEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.ollama_base_url,
        )
        self._store = Chroma(
            collection_name=settings.collection_name,
            persist_directory=str(settings.vector_store_dir),
            embedding_function=self._embeddings,
        )

    @property
    def store(self) -> Chroma:
        return self._store

    @property
    def vector_store_path(self) -> Path:
        return self._settings.vector_store_dir

    def add_documents(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        self._store.add_documents(docs)
        return len(docs)
