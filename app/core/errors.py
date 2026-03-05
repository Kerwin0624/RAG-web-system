class RAGError(Exception):
    """Base exception for RAG application."""


class IngestError(RAGError):
    """Raised when ingestion fails."""


class RetrievalError(RAGError):
    """Raised when retrieval fails."""


class GenerationError(RAGError):
    """Raised when generation fails."""


class EvaluationError(RAGError):
    """Raised when evaluation fails."""
