import time
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag_engine.config import settings
from rag_engine.core.embeddings import embed_texts
from rag_engine.utils.logger import get_logger


logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class VectorStoreManager:
    """
    Qdrant Cloud vector store.

    Uses the same:
        Gemini embeddings
        3072 dimensions
        Qdrant collection
        as the existing working RAG implementation.
    """

    def __init__(self) -> None:
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            try:
                self._client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=30,
                )

                self._client.get_collections()

                logger.info(
                    "Connected to Qdrant Cloud."
                )

            except Exception as exc:
                logger.error(
                    f"Qdrant connection failed: {exc}"
                )

                raise VectorStoreError(
                    f"Failed to connect to Qdrant: {exc}"
                ) from exc

        return self._client

    # ============================================================
    # HEALTH
    # ============================================================

    def health_check(self) -> dict:
        try:
            collections = self.client.get_collections()

            names = [
                collection.name
                for collection in collections.collections
            ]

            exists = (
                settings.qdrant_collection_name
                in names
            )

            count = 0

            if exists:
                info = self.client.get_collection(
                    settings.qdrant_collection_name
                )

                count = info.points_count or 0

            return {
                "connected": True,
                "collection_exists": exists,
                "documents_indexed": count,
            }

        except Exception as exc:
            logger.error(
                f"Qdrant health check failed: {exc}"
            )

            return {
                "connected": False,
                "collection_exists": False,
                "documents_indexed": 0,
                "error": str(exc),
            }

    # ============================================================
    # INDEX
    # ============================================================

    def index_documents(
        self,
        chunks,
    ) -> tuple[list[str], float]:

        start = time.perf_counter()

        try:
            texts = [
                chunk.page_content
                for chunk in chunks
            ]

            vectors = embed_texts(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
            )

            ids: list[str] = []

            points = []

            for index, (chunk, vector) in enumerate(
                zip(chunks, vectors)
            ):
                source = chunk.metadata.get(
                    "source_filename",
                    "unknown",
                )

                chunk_index = chunk.metadata.get(
                    "chunk_index",
                    index,
                )

                import uuid

                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{source}:{chunk_index}",
                    )
                )

                ids.append(point_id)

                payload = {
                    "text": chunk.page_content,
                    "source": source,
                    "chunk_index": chunk_index,
                    **chunk.metadata,
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=(
                    settings.qdrant_collection_name
                ),
                points=points,
                wait=True,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            logger.info(
                f"Indexed {len(points)} chunks "
                f"in {elapsed_ms:.1f}ms"
            )

            return ids, elapsed_ms

        except Exception as exc:
            logger.error(
                f"Indexing failed: {exc}"
            )

            raise VectorStoreError(
                f"Failed to index documents: {exc}"
            ) from exc

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[dict, float]]:

        start = time.perf_counter()

        top_k = (
            top_k
            if top_k is not None
            else settings.retrieval_top_k
        )

        score_threshold = (
            score_threshold
            if score_threshold is not None
            else settings.similarity_threshold
        )

        try:
            query_vector = embed_texts(
                [query],
                task_type="RETRIEVAL_QUERY",
            )[0]

            response = self.client.query_points(
                collection_name=(
                    settings.qdrant_collection_name
                ),
                query=query_vector,
                with_payload=True,
                limit=top_k,
                score_threshold=score_threshold,
            )

            results = []

            for point in response.points:

                payload = (
                    point.payload
                    or {}
                )

                results.append(
                    (
                        payload,
                        float(point.score),
                    )
                )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            logger.info(
                f"Search returned "
                f"{len(results)} results "
                f"in {elapsed_ms:.1f}ms"
            )

            return results

        except Exception as exc:
            logger.error(
                f"Search failed: {exc}"
            )

            raise VectorStoreError(
                f"Search failed: {exc}"
            ) from exc


vector_store_manager = VectorStoreManager()