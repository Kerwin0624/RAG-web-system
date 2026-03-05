import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = Field(default="RAG Web System", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    vector_store_dir: Path = Field(default=Path("./vector_store"), alias="VECTOR_STORE_DIR")
    collection_name: str = Field(default="default_collection", alias="COLLECTION_NAME")

    embedding_model_name: str = Field(
        default="bge-m3", alias="EMBEDDING_MODEL_NAME"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )

    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    search_top_k: int = Field(default=4, alias="SEARCH_TOP_K")
    search_score_threshold: float = Field(default=0.25, alias="SEARCH_SCORE_THRESHOLD")

    pdf_loader: str = Field(default="pypdf", alias="PDF_LOADER")
    vector_weight: float = Field(default=0.7, alias="VECTOR_WEIGHT")
    bm25_weight: float = Field(default=0.3, alias="BM25_WEIGHT")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")

    llm_base_url: str = Field(alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(alias="LLM_MODEL")
    llm_timeout: int = Field(default=60, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    default_temperature: float = Field(default=0.2, alias="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=512, alias="DEFAULT_MAX_TOKENS")
    default_top_p: float = Field(default=1.0, alias="DEFAULT_TOP_P")

    eval_batch_size: int = Field(default=8, alias="EVAL_BATCH_SIZE")

    @model_validator(mode="after")
    def _resolve_llm_api_key(self) -> "Settings":
        env_key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if env_key:
            self.llm_api_key = env_key
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    return settings
