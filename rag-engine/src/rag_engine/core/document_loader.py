import hashlib
import time
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from rag_engine.config import settings
from rag_engine.utils.logger import get_logger


logger = get_logger(__name__)


class DocumentLoadError(Exception):
    """Raised when document loading fails."""


class DocumentProcessor:

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):

        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else settings.chunk_size
        )

        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.chunk_overlap
        )

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )
        )

    def validate_file(
        self,
        filename: str,
        file_size_bytes: int,
    ) -> None:

        extension = (
            Path(filename)
            .suffix
            .lower()
        )

        if extension not in settings.supported_formats:
            raise DocumentLoadError(
                f"Unsupported format: {extension}"
            )

        max_bytes = (
            settings.max_upload_size_mb
            * 1024
            * 1024
        )

        if file_size_bytes > max_bytes:
            raise DocumentLoadError(
                f"File too large: "
                f"{file_size_bytes / 1024 / 1024:.1f}MB"
            )

    def compute_hash(
        self,
        content: bytes,
    ) -> str:

        return hashlib.sha256(
            content
        ).hexdigest()

    def load_and_chunk(
        self,
        file_path: str,
        filename: str,
    ) -> tuple[list[Document], float]:

        start = time.perf_counter()

        extension = (
            Path(filename)
            .suffix
            .lower()
        )

        loader_cls = (
            self.LOADER_MAP.get(extension)
        )

        if not loader_cls:
            raise DocumentLoadError(
                f"No loader for {extension}"
            )

        try:
            documents = loader_cls(
                file_path
            ).load()

        except Exception as exc:
            logger.error(
                f"Failed to load {filename}: {exc}"
            )

            raise DocumentLoadError(
                f"Failed to parse {filename}: {exc}"
            ) from exc

        if not documents:
            raise DocumentLoadError(
                f"No content extracted from {filename}"
            )

        for document in documents:

            document.metadata[
                "source_filename"
            ] = filename

        chunks = (
            self.splitter.split_documents(
                documents
            )
        )

        for index, chunk in enumerate(chunks):

            chunk.metadata[
                "chunk_index"
            ] = index

            chunk.metadata[
                "total_chunks"
            ] = len(chunks)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            f"Processed {filename}: "
            f"{len(documents)} pages → "
            f"{len(chunks)} chunks "
            f"in {elapsed_ms:.1f}ms"
        )

        return chunks, elapsed_ms