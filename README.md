# Production-Grade RAG Engine

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-f90050?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=flat&logo=google)](https://ai.google.dev/)

A production-focused Retrieval-Augmented Generation (RAG) application for uploading PDF documents, indexing their content, retrieving relevant context, and generating grounded answers using Google Gemini.

The project is designed with a **local-first development workflow** while supporting cloud-based LLM and embedding providers for deployment.

---

## 🚀 Features

- 📄 PDF document ingestion
- ✂️ Intelligent document chunking
- 🔎 Vector similarity search with Qdrant
- 🧠 Google Gemini for LLM generation
- 🔢 Gemini embeddings for semantic retrieval
- ⚡ FastAPI backend
- 🎨 Streamlit frontend
- 🔄 Inngest workflow support
- 🛡️ Evidence-based RAG responses
- 📚 Source-aware retrieval
- 🔁 Deterministic document IDs for idempotent ingestion
- 💾 Local Qdrant storage for development
- 🧩 Multi-provider architecture
- 🛠️ Graceful local fallback when model services are unavailable

---

# 🏗️ Architecture

```text
                    ┌──────────────────┐
                    │    PDF Upload    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   PDF Reader     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Text Chunking    │
                    │ SentenceSplitter │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Gemini Embeddings│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Qdrant       │
                    │  Vector Storage  │
                    └────────┬─────────┘
                             │
                  User Question
                             │
                             ▼
                    ┌──────────────────┐
                    │ Query Embedding  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Top-K Retrieval  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Gemini LLM       │
                    │ Grounded Answer  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Streamlit UI     │
                    └──────────────────┘
🧠 RAG Pipeline

The application follows the standard Retrieval-Augmented Generation pipeline:

1. Upload

A user uploads a PDF through the Streamlit interface.

2. Parse

The PDF is processed using LlamaIndex's PDFReader.

3. Chunk

The extracted text is split into overlapping chunks.

Chunk Size:     1000 tokens/characters
Chunk Overlap:  200
4. Embed

Each chunk is converted into a vector using Gemini embeddings.

5. Store

Embeddings and document metadata are stored in Qdrant.

6. Retrieve

When the user asks a question:

Question
   ↓
Embedding
   ↓
Qdrant Similarity Search
   ↓
Top-K Relevant Chunks
7. Generate

The retrieved context is provided to Gemini with a strict grounding instruction.

The model is instructed to avoid hallucinating information that does not exist in the retrieved document.

📁 Project Structure
Production-grade-RAG-main/
│
├── main.py
│   └── FastAPI application + RAG API + Inngest functions
│
├── streamlit_app.py
│   └── Streamlit frontend
│
├── data_loader.py
│   └── PDF loading, text normalization, chunking and embeddings
│
├── vector_db.py
│   └── Qdrant vector database integration
│
├── custom_types.py
│   └── Pydantic data models
│
├── pyproject.toml
│   └── Python project configuration and dependencies
│
├── uv.lock
│   └── Locked dependency versions
│
├── README.md
│   └── Project documentation
│
├── doc.md
│   └── Detailed technical documentation
│
├── .env
│   └── Local environment configuration
│
├── uploads/
│   └── Uploaded PDF files
│
├── qdrant_storage/
│   └── Local Qdrant vector storage
│
├── rag-engine/
│   └── Packaged implementation
│
└── .gitignore
⚙️ Requirements
Python 3.12+
Google Gemini API key
Git
Windows/Linux/macOS

For local development, Qdrant can run using embedded/local storage.

🚀 Quick Start
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Production-grade-RAG.git

cd Production-grade-RAG
2. Create virtual environment
Windows PowerShell
python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

Then:

.\.venv\Scripts\Activate.ps1
📦 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -e .
🔑 4. Configure Gemini API

Create a .env file in the repository root.

LLM_PROVIDER=gemini
EMBED_PROVIDER=gemini

GEMINI_API_KEY=your_gemini_api_key

GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-2

EMBED_DIM=768

QDRANT_PATH=./qdrant_storage

INNGEST_ENABLED=false
INNGEST_DEV=0

Replace:

your_gemini_api_key

with your actual Gemini API key.

Important

Never commit .env to GitHub.

Your .gitignore should contain:

.env
.venv/
__pycache__/
qdrant_storage/
uploads/
*.pyc
▶️ 5. Start the Streamlit application
python -m streamlit run streamlit_app.py

The application will be available at:

http://127.0.0.1:8501
📄 6. Upload a PDF

Open the Streamlit application and upload a PDF.

Example:

Enterprise RAG Test Document.pdf

The application will:

PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Gemini Embeddings
 ↓
Qdrant

You should see a message similar to:

Ingested locally: X chunks
💬 7. Ask Questions

After ingestion, ask questions about the uploaded document.

Example:

What technologies are mentioned in the document?

or:

What is the candidate's experience with Python?

The system retrieves relevant chunks and sends only that context to Gemini.

🔌 API Endpoints

The FastAPI application exposes local endpoints.

Start FastAPI:

python -m uvicorn main:app --host 127.0.0.1 --port 8000

API:

http://127.0.0.1:8000
PDF Ingestion
POST
/api/local-ingest

Example request:

{
  "pdf_path": "uploads/document.pdf",
  "source_id": "document.pdf"
}

Example response:

{
  "ingested": 12,
  "source_id": "document.pdf"
}
Context Retrieval
POST
/api/local-query-context

Example:

{
  "question": "What skills are mentioned?",
  "top_k": 5,
  "source_hint": "document.pdf"
}
AI Answer
POST
/api/local-query-ai

Example:

{
  "question": "What technologies does the candidate know?",
  "top_k": 5,
  "source_hint": "document.pdf"
}
🧩 Configuration

The application supports multiple providers.

LLM_PROVIDER=gemini
EMBED_PROVIDER=gemini

Supported providers:

LLM_PROVIDER
├── openai
├── gemini
├── claude
├── ollama
└── local

EMBED_PROVIDER
├── openai
├── gemini
├── ollama
└── local

For the recommended cloud deployment configuration:

LLM_PROVIDER=gemini
EMBED_PROVIDER=gemini
🤖 Gemini Configuration

The project uses Gemini for both:

LLM
GEMINI_MODEL=gemini-2.5-flash
Embeddings
GEMINI_EMBED_MODEL=gemini-embedding-2
API Key
GEMINI_API_KEY=your_api_key

The API key is loaded through environment variables and is never hard-coded into the source code.

🦙 Ollama Support

The application also supports Ollama for completely local development.

Example:

LLM_PROVIDER=ollama
EMBED_PROVIDER=ollama

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_EMBED_MODEL=nomic-embed-text

EMBED_DIM=768

Install the required models:

ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

Verify:

ollama list

Start Ollama:

ollama run qwen2.5-coder:7b

Ollama is recommended for local development because models run on the developer's machine.

For cloud deployment, a hosted model provider such as Gemini is generally easier than running Ollama inside a serverless environment.

🗄️ Qdrant

The project supports local Qdrant storage:

QDRANT_PATH=./qdrant_storage

This allows the application to work without requiring a separately hosted Qdrant server during local development.

For production deployments, use a persistent external Qdrant instance or another persistent vector database.

🔄 Inngest

The project supports Inngest for durable background workflows.

Local fallback mode:

INNGEST_ENABLED=false
INNGEST_DEV=0

Inngest mode can be enabled when a separate workflow infrastructure is configured.

The ingestion workflow is conceptually:

PDF Upload
    ↓
Inngest Event
    ↓
Load PDF
    ↓
Chunk Document
    ↓
Generate Embeddings
    ↓
Upsert to Qdrant

The query workflow:

User Question
    ↓
Generate Query Embedding
    ↓
Retrieve Context
    ↓
Generate Gemini Answer
    ↓
Return Sources + Answer
☁️ Deployment

The application can be deployed using a cloud architecture such as:

                    ┌───────────────┐
                    │   Streamlit   │
                    │   Frontend    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    FastAPI    │
                    │    Backend    │
                    └───────┬───────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
         Gemini          Qdrant         Inngest
          API           Cloud/DB       Workflows
Important deployment consideration

The local configuration:

QDRANT_PATH=./qdrant_storage

is suitable for development but should not be treated as durable production storage on ephemeral/serverless infrastructure.

For production:

Frontend
   ↓
Backend
   ↓
Gemini API
   ↓
Hosted Qdrant
🔐 Security

Never commit API keys.

Do not commit:

.env

Do not hard-code:

GEMINI_API_KEY = "..."

Use environment variables instead:

os.getenv("GEMINI_API_KEY")

For deployment platforms, configure the API key through their environment-variable/secrets settings.

🛡️ Grounded Generation

The application uses a strict RAG system prompt:

Answer ONLY from provided context.

If context does not contain the answer, reply:

I don't know based on provided documents.

Do not invent facts or skills.

This reduces hallucination and ensures responses remain grounded in retrieved document content.

⚡ Engineering Highlights
Multi-provider architecture

LLM and embedding providers can be switched using environment variables without rewriting the entire application.

Deterministic document IDs

Document chunks use deterministic UUIDs to support idempotent ingestion.

Source-aware retrieval

Retrieved chunks retain their source document information.

Local-first development

The application can run locally using:

Gemini
Ollama
Local fallback
Embedded Qdrant storage
Graceful degradation

If the configured LLM becomes unavailable, the application can fall back to evidence-based local responses.

Retrieval ranking

Retrieved chunks are additionally ranked using:

Semantic similarity
Keyword overlap
Skill-specific relevance signals
🧪 Testing

For the packaged implementation:

cd rag-engine

python -m pip install -e ".[dev]"

pytest

Run linting:

ruff check src tests
🐳 Docker

The packaged implementation contains Docker configuration.

From the rag-engine directory:

docker compose -f docker/docker-compose.yml up --build
🛠️ Troubleshooting
Gemini API error

Check:

GEMINI_API_KEY=your_key

and:

LLM_PROVIDER=gemini
EMBED_PROVIDER=gemini
Embedding dimension error

Make sure the configured dimension matches the embedding model:

EMBED_DIM=768

If the embedding model configuration changes, the Qdrant collection may need to be recreated.

Qdrant error

Check:

QDRANT_PATH=./qdrant_storage

Ensure the application has permission to write to the directory.

Ollama error

Check:

ollama list

Make sure the required models exist:

qwen2.5-coder:7b
nomic-embed-text

Also verify Ollama is running:

http://127.0.0.1:11434
Streamlit not starting

Run:

python -m streamlit run streamlit_app.py

instead of relying on the global streamlit command.

📌 Example Use Cases

This architecture can be adapted for:

Resume analysis
Enterprise document Q&A
Technical documentation search
Policy and compliance assistants
Research document analysis
Internal knowledge bases
Customer support knowledge systems
HR document assistants
Legal document retrieval
Financial document analysis
📈 Future Improvements

Potential production improvements include:

Hosted Qdrant
Authentication and authorization
Document-level access control
Streaming LLM responses
Async embedding generation
Batch embedding requests
Document versioning
Hybrid BM25 + vector retrieval
Cross-encoder reranking
Observability with LangSmith
Prometheus/Grafana monitoring
Dockerized deployment
CI/CD
Automated evaluation pipelines
Multi-user document isolation
📜 License

Add your preferred license to the repository, such as:

MIT

or:

Apache-2.0
👨‍💻 Author

Built as a production-oriented RAG engineering project focused on:

Python · FastAPI · Streamlit · Gemini · Embeddings · Qdrant · RAG · LLM Applications · AI Engineering