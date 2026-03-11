from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=list, description="File paths to ingest.")
    directory: str | None = Field(default=None, description="Directory path to ingest.")
    recursive: bool = Field(default=True, description="Whether to recursively scan directory.")
    pdf_loader: str = Field(default="pypdf", description="PDF loader: 'pypdf' or 'mineru'.")
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    chunk_separators: list[str] | None = Field(default=None, description="Custom chunk separators.")

    @model_validator(mode="after")
    def validate_input_sources(self) -> "IngestRequest":
        if not self.paths and not self.directory:
            raise ValueError("paths 或 directory 至少提供一个。")
        return self


class IngestResponse(BaseModel):
    chunks_indexed: int
    sources: list[str]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    search_type: Literal["similarity", "bm25", "hybrid", "mmr"] = "similarity"
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    reranker_enabled: bool = Field(default=False)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=32, le=4096)
    top_p: float = Field(default=1.0, ge=0.1, le=1.0)
    model: str | None = Field(default=None, description="LLM model name override.")


class RetrievedChunk(BaseModel):
    content: str
    source: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[RetrievedChunk]
    elapsed_ms: int
    from_cache: bool = False


class EvalSample(BaseModel):
    question: str
    answer: str
    ground_truth: str
    contexts: list[str]


class EvalRequest(BaseModel):
    samples: list[EvalSample]
    save_report: bool = False
    report_dir: str = ""


class EvalResponse(BaseModel):
    metrics: dict[str, Any]
    report_files: dict[str, str] | None = None
