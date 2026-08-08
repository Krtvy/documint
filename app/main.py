"""
DocuMint - AI-Powered Document Q&A System
Main FastAPI Application

Endpoints:
- POST /upload - Upload and process a document
- POST /query - Ask a question about a document
- GET /documents - List all uploaded documents
- GET /documents/{id} - Get document details
- DELETE /documents/{id} - Delete a document
- GET /health - Health check
"""

import os
import time
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from app.config import get_settings
from app.database import (
    get_db, init_db, create_document, add_chunks,
    get_document, get_all_documents, delete_document
)
from app.extraction import extract_text, get_file_type, is_supported_file
from app.chunking import create_chunks
from app.embeddings import generate_embeddings, preload_model
from app.rag import query_document
from app.models import (
    QueryRequest, QueryResponse, DocumentResponse, DocumentListResponse,
    UploadResponse, HealthResponse
)

settings = get_settings()


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown events.
    - Initialize database on startup
    - Preload embedding model
    """
    print("🚀 Starting DocuMint...")
    
    # Initialize database tables
    init_db()
    
    # Preload embedding model (avoids delay on first request)
    preload_model()
    
    print("✅ DocuMint is ready!")
    
    yield
    
    print("👋 Shutting down DocuMint...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="DocuMint",
    description="AI-Powered Document Q&A System using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint."""
    return {
        "message": "Welcome to DocuMint! 📄🤖",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload",
            "query": "POST /query",
            "documents": "GET /documents",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    Verifies database connection and model availability.
    """
    try:
        # SQLAlchemy 2.0 requires raw SQL to be wrapped in text().
        db.execute(text("SELECT 1"))
        db_status = "connected"
        status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
        status = "degraded"

    return HealthResponse(
        status=status,
        database=db_status,
        embedding_model=settings.embedding_model
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document.
    
    Supported formats: PDF, DOCX, PPTX, XLSX
    
    Processing steps:
    1. Validate file type
    2. Save file to disk
    3. Extract text from document
    4. Split text into chunks
    5. Generate embeddings for chunks
    6. Store in database
    """
    start_time = time.time()
    
    # Validate file type
    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: PDF, DOCX, PPTX, XLSX"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    max_size = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Save file to disk
    file_type = get_file_type(file.filename)
    file_path = os.path.join(settings.upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        # Step 1: Extract text
        print(f"📄 Extracting text from {file.filename}...")
        text = extract_text(file_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        
        # Step 2: Create chunks
        print(f"✂️ Splitting text into chunks...")
        chunks = create_chunks(text)
        print(f"   Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        print(f"🧮 Generating embeddings...")
        embeddings = generate_embeddings(chunks)
        
        # Step 4: Create document record
        doc = create_document(
            db=db,
            filename=file.filename,
            file_type=file_type,
            file_path=file_path,
            file_size=file_size
        )
        
        # Step 5: Store chunks with embeddings
        print(f"💾 Storing chunks in database...")
        chunk_data = [
            {'content': chunk, 'embedding': embedding}
            for chunk, embedding in zip(chunks, embeddings)
        ]
        add_chunks(db, doc.id, chunk_data)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Document processed in {processing_time:.2f}s")
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            document_id=doc.id,
            filename=file.filename,
            file_type=file_type,
            total_chunks=len(chunks),
            processing_time_seconds=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query_document_endpoint(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Ask a question about a document.
    
    Uses RAG (Retrieval Augmented Generation):
    1. Generate embedding for the question
    2. Find similar chunks in the document
    3. Send chunks + question to Claude
    4. Return answer with source citations
    """
    # Get document
    doc = get_document(db, request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Run RAG pipeline
        result = query_document(
            db=db,
            document_id=request.document_id,
            query=request.question,
            document_name=doc.filename,
            top_k=request.top_k
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            model=result['model'],
            chunks_used=result['chunks_used'],
            document_id=request.document_id,
            question=request.question
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents(db: Session = Depends(get_db)):
    """List all uploaded documents."""
    docs = get_all_documents(db)
    
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                total_chunks=doc.total_chunks,
                created_at=doc.created_at
            )
            for doc in docs
        ],
        total=len(docs)
    )


@app.get("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document_details(document_id: int, db: Session = Depends(get_db)):
    """Get details of a specific document."""
    doc = get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        total_chunks=doc.total_chunks,
        created_at=doc.created_at
    )


@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document_endpoint(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document and all its chunks.
    Also removes the file from disk.
    """
    doc = get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file from disk
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    
    # Delete from database
    delete_document(db, document_id)
    
    return {"message": f"Document '{doc.filename}' deleted successfully"}


# =============================================================================
# Run with: uvicorn app.main:app --reload
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
