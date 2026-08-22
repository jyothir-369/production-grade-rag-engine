from datetime import datetime

from pydantic import BaseModel, Field


class IngestDocumentRequest(BaseModel):
    pdf_path: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)


class RAGChunkAndSrc(BaseModel):
    text: str
    source: str
    score: float | None = None
    chunk_index: int | None = None


class RAGSearchResult(BaseModel):
    records: list[RAGChunkAndSrc]
    sources: list[str]
    num_contexts: int


class RAGQueryResult(BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int


class RAGUpsertResult(BaseModel):
    source_id: str
    chunks_indexed: int
    processing_time_ms: float


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class SourceDocument(BaseModel):
    content: str
    filename: str
    page: int | None = None
    similarity_score: float
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query_time_ms: float
    tokens_used: int | None = None
    model: str
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class DocumentStatus:
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_created: int
    processing_time_ms: float
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    documents_indexed: int
    uptime_seconds: float
    version: str