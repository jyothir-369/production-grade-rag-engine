from openai import OpenAI

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv

import os
import requests
import hashlib
import re


load_dotenv(override=True)


# ============================================================
# CONFIGURATION
# ============================================================

EMBED_PROVIDER = os.getenv(
    "EMBED_PROVIDER",
    "gemini",
).lower()

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "text-embedding-3-large",
)

GEMINI_EMBED_MODEL = os.getenv(
    "GEMINI_EMBED_MODEL",
    "gemini-embedding-001",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text",
)


# ============================================================
# OLLAMA AUTHENTICATION
# ============================================================

def _ollama_headers() -> dict[str, str]:
    api_key = os.getenv("OLLAMA_API_KEY")

    if not api_key:
        return {}

    return {
        "Authorization": f"Bearer {api_key}"
    }


# ============================================================
# EMBEDDING DIMENSIONS
# ============================================================

def _default_embed_dim(provider: str) -> int:
    if provider == "gemini":
        return 3072

    if provider == "ollama":
        return 768

    if provider == "local":
        return 768

    return 3072


EMBED_DIM = int(
    os.getenv(
        "EMBED_DIM",
        str(_default_embed_dim(EMBED_PROVIDER)),
    )
)


# ============================================================
# OPENAI CLIENT
# ============================================================

def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for "
            "EMBED_PROVIDER=openai"
        )

    return OpenAI(api_key=api_key)


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

def _get_embed_config() -> tuple[str, str, str, str, int]:

    provider = os.getenv(
        "EMBED_PROVIDER",
        EMBED_PROVIDER,
    ).lower()

    model = os.getenv(
        "EMBED_MODEL",
        EMBED_MODEL,
    )

    ollama_base = os.getenv(
        "OLLAMA_BASE_URL",
        OLLAMA_BASE_URL,
    )

    ollama_model = os.getenv(
        "OLLAMA_EMBED_MODEL",
        OLLAMA_EMBED_MODEL,
    )

    dim = int(
        os.getenv(
            "EMBED_DIM",
            str(_default_embed_dim(provider)),
        )
    )

    return (
        provider,
        model,
        ollama_base,
        ollama_model,
        dim,
    )


# ============================================================
# DETERMINISTIC LOCAL FALLBACK
# ============================================================

def _hash_embedding(
    text: str,
    dim: int,
) -> list[float]:
    """
    Deterministic fallback embedding used when
    a local embedding provider is selected.
    """

    seed = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    out: list[float] = []
    counter = 0

    while len(out) < dim:

        block = hashlib.sha256(
            seed + counter.to_bytes(4, "big")
        ).digest()

        for i in range(0, len(block), 4):

            if len(out) >= dim:
                break

            value = int.from_bytes(
                block[i:i + 4],
                "big",
            )

            out.append(
                (value / 4294967295.0) * 2.0 - 1.0
            )

        counter += 1

    return out


# ============================================================
# TEXT CHUNKING
# ============================================================

splitter = SentenceSplitter(
    chunk_size=350,
    chunk_overlap=50,
)


# ============================================================
# PDF TEXT NORMALIZATION
# ============================================================

def _normalize_pdf_text(text: str) -> str:

    if not text:
        return ""

    cleaned = (
        text
        .replace("\u00a0", " ")
        .replace("\ufeff", " ")
    )

    # Fix PDF text where individual characters
    # are separated by spaces.
    cleaned = re.sub(
        r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b",
        lambda m: m.group(0).replace(" ", ""),
        cleaned,
    )

    # Normalize spaces and tabs.
    cleaned = re.sub(
        r"[ \t]+",
        " ",
        cleaned,
    )

    # Collapse excessive blank lines.
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


# ============================================================
# LOAD AND CHUNK PDF
# ============================================================

def load_and_chunk_pdf(path: str) -> list[str]:

    docs = PDFReader().load_data(
        file=path
    )

    texts = [
        _normalize_pdf_text(d.text)
        for d in docs
        if getattr(d, "text", None)
    ]

    chunks: list[str] = []

    for text in texts:

        if not text:
            continue

        chunks.extend(
            splitter.split_text(text)
        )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_texts(
    texts: list[str],
) -> list[list[float]]:

    if not texts:
        return []

    (
        provider,
        model,
        ollama_base,
        ollama_model,
        dim,
    ) = _get_embed_config()


    # ========================================================
    # OPENAI
    # ========================================================

    if provider == "openai":

        response = _openai_client().embeddings.create(
            model=model,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]


    # ========================================================
    # GEMINI
    # ========================================================

    if provider == "gemini":

        try:
            from google import genai
            from google.genai import types

        except ImportError as exc:

            raise RuntimeError(
                "Gemini embeddings require the "
                "google-genai package. "
                "Install it with: "
                "python -m pip install -U google-genai"
            ) from exc


        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not api_key:

            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY "
                "is required for "
                "EMBED_PROVIDER=gemini"
            )


        client = genai.Client(
            api_key=api_key
        )


        try:

            result = client.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=dim,
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )

        except Exception as exc:

            raise RuntimeError(
                f"Gemini embedding request failed: {exc}"
            ) from exc


        if not result.embeddings:

            raise RuntimeError(
                "Gemini returned no embeddings."
            )


        vectors = [
            embedding.values
            for embedding in result.embeddings
        ]


        # Validate embedding dimensions before
        # sending vectors to Qdrant.
        for index, vector in enumerate(vectors):

            if len(vector) != dim:

                raise RuntimeError(
                    "Gemini returned an unexpected "
                    f"embedding dimension for item {index}: "
                    f"expected {dim}, got {len(vector)}"
                )


        return vectors


    # ========================================================
    # OLLAMA
    # ========================================================

    if provider == "ollama":

        vectors: list[list[float]] = []

        for text in texts:

            try:

                response = requests.post(
                    f"{ollama_base}/api/embeddings",
                    headers=_ollama_headers(),
                    json={
                        "model": ollama_model,
                        "prompt": text,
                    },
                    timeout=60,
                )

                response.raise_for_status()

                embedding = response.json().get(
                    "embedding"
                )

                if not embedding:

                    raise RuntimeError(
                        "Ollama returned an empty embedding."
                    )

                vectors.append(embedding)

            except Exception as exc:

                print(
                    f"Warning: Ollama embedding failed: {exc}. "
                    "Using deterministic fallback."
                )

                vectors.append(
                    _hash_embedding(
                        text,
                        dim,
                    )
                )

        return vectors


    # ========================================================
    # LOCAL HASH FALLBACK
    # ========================================================

    if provider == "local":

        return [
            _hash_embedding(
                text,
                dim,
            )
            for text in texts
        ]


    # ========================================================
    # INVALID PROVIDER
    # ========================================================

    raise ValueError(
        "Unsupported EMBED_PROVIDER: "
        f"{provider}. "
        "Use one of: openai, gemini, ollama, local."
    )