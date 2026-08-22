from fastapi import APIRouter, HTTPException

from rag_engine.core.rag_chain import RAGChain
from rag_engine.models.schemas import (
    IngestDocumentRequest,
    RAGUpsertResult,
)


router = APIRouter()

chain = RAGChain()


@router.post(
    "/ingest",
    response_model=RAGUpsertResult,
)
def ingest_document(
    payload: IngestDocumentRequest,
) -> RAGUpsertResult:

    try:
        return chain.ingest_pdf(
            pdf_path=payload.pdf_path,
            source_id=payload.source_id,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc