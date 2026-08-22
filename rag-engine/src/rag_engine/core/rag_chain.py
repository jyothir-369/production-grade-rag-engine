import time

from google import genai

from rag_engine.config import settings
from rag_engine.core.document_loader import (
    DocumentLoadError,
    DocumentProcessor,
)
from rag_engine.core.vector_store import (
    VectorStoreError,
    vector_store_manager,
)
from rag_engine.models.schemas import (
    IngestDocumentRequest,
    QueryRequest,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
    RAGChunkAndSrc,
)


class RAGChain:

    def __init__(self):

        self.document_processor = (
            DocumentProcessor()
        )

        self.vector_store = (
            vector_store_manager
        )

        self.llm_client = genai.Client(
            api_key=settings.gemini_api_key
        )

    # ============================================================
    # INGEST
    # ============================================================

    def ingest_pdf(
        self,
        pdf_path: str,
        source_id: str,
    ) -> RAGUpsertResult:

        start = time.perf_counter()

        chunks, _ = (
            self.document_processor.load_and_chunk(
                pdf_path,
                source_id,
            )
        )

        _, indexing_time = (
            self.vector_store.index_documents(
                chunks
            )
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        return RAGUpsertResult(
            source_id=source_id,
            chunks_indexed=len(chunks),
            processing_time_ms=(
                elapsed_ms + indexing_time
            ),
        )

    # ============================================================
    # RETRIEVE
    # ============================================================

    def retrieve(
        self,
        question: str,
        top_k: int,
        similarity_threshold: float,
    ) -> RAGSearchResult:

        results = self.vector_store.search(
            query=question,
            top_k=top_k,
            score_threshold=similarity_threshold,
        )

        records = []

        sources = set()

        for payload, score in results:

            source = payload.get(
                "source",
                "unknown",
            )

            sources.add(source)

            records.append(
                RAGChunkAndSrc(
                    text=payload.get(
                        "text",
                        "",
                    ),
                    source=source,
                    score=score,
                    chunk_index=payload.get(
                        "chunk_index"
                    ),
                )
            )

        return RAGSearchResult(
            records=records,
            sources=sorted(sources),
            num_contexts=len(records),
        )

    # ============================================================
    # GENERATE
    # ============================================================

    def _generate_answer(
        self,
        question: str,
        contexts: list[str],
    ) -> str:

        context_text = "\n\n---\n\n".join(
            contexts
        )

        prompt = f"""
You are a grounded RAG assistant.

Answer the user's question using ONLY
the provided context.

If the answer cannot be found in the
context, clearly say that the information
is not available in the provided documents.

Do not invent facts.

Context:
{context_text}

Question:
{question}

Answer:
"""

        response = (
            self.llm_client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty answer."
            )

        return response.text.strip()

    # ============================================================
    # QUERY
    # ============================================================

    def query(
        self,
        request: QueryRequest,
    ) -> RAGQueryResult:

        search_result = self.retrieve(
            question=request.question,
            top_k=request.top_k,
            similarity_threshold=(
                request.similarity_threshold
            ),
        )

        if not search_result.records:

            return RAGQueryResult(
                answer=(
                    "I could not find relevant "
                    "information in the indexed documents."
                ),
                sources=[],
                num_contexts=0,
            )

        contexts = [
            record.text
            for record in search_result.records
        ]

        answer = self._generate_answer(
            question=request.question,
            contexts=contexts,
        )

        return RAGQueryResult(
            answer=answer,
            sources=search_result.sources,
            num_contexts=search_result.num_contexts,
        )