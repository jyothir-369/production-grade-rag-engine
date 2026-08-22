from fastapi import APIRouter, HTTPException

from rag_engine.core.rag_chain import (
    RAGChain,
)
from rag_engine.core.vector_store import (
    VectorStoreError,
)
from rag_engine.models.schemas import (
    QueryRequest,
    RAGQueryResult,
)


router = APIRouter()

chain = RAGChain()


@router.post(
    "/query",
    response_model=RAGQueryResult,
)
def query_documents(
    request: QueryRequest,
) -> RAGQueryResult:

    try:
        return chain.query(request)

    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc