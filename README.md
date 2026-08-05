# 🌿 DocuMint - AI-Powered Document Q&A System

A production-ready RAG (Retrieval Augmented Generation) system that lets you upload documents and ask questions about them using natural language.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)
![pgvector](https://img.shields.io/badge/pgvector-0.8.1-orange.svg)

## ✨ Features

- **Multi-format Support**: PDF, Word (DOCX), PowerPoint (PPTX), Excel (XLSX)
- **Intelligent Chunking**: LangChain's recursive text splitter for semantic coherence
- **Fast Vector Search**: PostgreSQL + pgvector with HNSW indexing
- **AI-Powered Answers**: Claude API for natural language responses
- **Source Citations**: Every answer includes references to source chunks

## 🏗️ Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Upload    │────▶│   Extract    │────▶│   Chunk Text    │
│  Document   │     │    Text      │     │  (LangChain)    │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Return    │◀────│   Generate   │◀────│    Generate     │
│   Answer    │     │   (Claude)   │     │   Embeddings    │
└─────────────┘     └──────────────┘     └────────┬────────┘
       ▲                   ▲                      │
       │                   │                      ▼
       │            ┌──────────────┐     ┌─────────────────┐
       │            │   Retrieve   │◀────│  Store in DB    │
       └────────────│   Chunks     │     │   (pgvector)    │
                    └──────────────┘     └─────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| PDF Extraction | pypdf |
| Word Extraction | python-docx |
| PPT Extraction | python-pptx |
| Excel Extraction | openpyxl |
| Text Chunking | LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | PostgreSQL + pgvector |
| LLM | Anthropic Claude |

## 📁 Project Structure
```
documint/
├── app/
│   ├── __init__.py       # Package initialization
│   ├── main.py           # FastAPI app & endpoints
│   ├── config.py         # Configuration settings
│   ├── database.py       # PostgreSQL + pgvector operations
│   ├── extraction.py     # Document text extraction
│   ├── chunking.py       # Text splitting with LangChain
│   ├── embeddings.py     # Sentence-transformers embeddings
│   ├── rag.py            # RAG pipeline + Claude integration
│   └── models.py         # Pydantic models
├── uploads/              # Uploaded documents
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 16 with pgvector extension
- Anthropic API key

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/documint.git
cd documint

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL database
psql -U postgres -c "CREATE DATABASE documint;"
psql -U postgres -d documint -c "CREATE EXTENSION vector;"

# Configure environment
cp .env.example .env
# Edit .env with your database password and Anthropic API key

# Run the server
uvicorn app.main:app --reload
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/upload` | Upload document |
| POST | `/query` | Ask question about document |
| GET | `/documents` | List all documents |
| GET | `/documents/{id}` | Get document details |
| DELETE | `/documents/{id}` | Delete document |

### Usage Examples

**Upload a document:**
```bash
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"document_id": 1, "question": "What is this document about?"}'
```

## 📊 API Response Examples

**Upload Response:**
```json
{
  "message": "Document uploaded and processed successfully",
  "document_id": 1,
  "filename": "document.pdf",
  "file_type": "pdf",
  "total_chunks": 42,
  "processing_time_seconds": 3.25
}
```

**Query Response:**
```json
{
  "answer": "Based on the document...",
  "sources": [
    {"chunk_index": 3, "preview": "The document discusses..."}
  ],
  "model": "claude-sonnet-4-20250514",
  "chunks_used": 5,
  "document_id": 1,
  "question": "What is this document about?"
}
```

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `ANTHROPIC_API_KEY` | - | Your Anthropic API key |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence-transformers model |

## 🛣️ Future Enhancements

- [ ] Async processing with Celery + Redis
- [ ] Multi-document queries
- [ ] Chat history and follow-up questions
- [ ] Web UI frontend
- [ ] Authentication & user management
- [ ] Docker containerization

## 📄 License

MIT License

## 👨‍💻 Author

**Kartavya Joshi**
- GitHub: [@Krtvy](https://github.com/Krtvy)
- Email: kartavvyajoshi@gmail.com
