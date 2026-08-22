from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are loaded from .env.
    Secrets are never hardcoded.
    """

    # ============================================================
    # LLM
    # ============================================================

    llm_provider: str = "gemini"
    gemini_api_key: str = Field(..., description="Gemini API key")
    gemini_model: str = "gemini-2.5-flash"

    # ============================================================
    # EMBEDDINGS
    # ============================================================

    embedding_provider: EmbeddingProvider = EmbeddingProvider.GEMINI

    gemini_embed_model: str = "gemini-embedding-001"

    embedding_dimensions: int = Field(
        default=3072,
        ge=1,
    )

    # ============================================================
    # QDRANT
    # ============================================================

    qdrant_url: str = Field(...)
    qdrant_api_key: str = Field(...)

    qdrant_collection_name: str = "docs"

    # ============================================================
    # DOCUMENT PROCESSING
    # ============================================================

    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=5000,
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
    )

    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=200,
    )

    supported_formats: list[str] = [
        ".pdf",
        ".txt",
        ".md",
    ]

    # ============================================================
    # RETRIEVAL
    # ============================================================

    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    # ============================================================
    # API
    # ============================================================

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    log_level: str = "INFO"

    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()