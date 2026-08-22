from google import genai
from google.genai import types

from rag_engine.config import settings


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Generate Gemini embeddings.

    The existing Qdrant collection uses 3072-dimensional
    Gemini embeddings.
    """

    if not texts:
        return []

    client = genai.Client(
        api_key=settings.gemini_api_key
    )

    response = client.models.embed_content(
        model=settings.gemini_embed_model,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimensions,
            task_type=task_type,
        ),
    )

    if not response.embeddings:
        raise RuntimeError(
            "Gemini returned no embeddings."
        )

    vectors = [
        embedding.values
        for embedding in response.embeddings
    ]

    for vector in vectors:
        if len(vector) != settings.embedding_dimensions:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected {settings.embedding_dimensions}, "
                f"got {len(vector)}"
            )

    return vectors