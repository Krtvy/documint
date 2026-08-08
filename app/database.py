"""
DocuMint Database Operations
Handles PostgreSQL + pgvector for document and embedding storage.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import List, Optional
import numpy as np

from app.config import get_settings

settings = get_settings()

# SQLAlchemy setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# =============================================================================
# Database Models
# =============================================================================

class Document(Base):
    """Stores uploaded document metadata."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, pptx, xlsx
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)  # bytes
    total_chunks = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Stores document chunks with their embeddings."""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order of chunk in document
    content = Column(Text, nullable=False)  # The actual text
    embedding = Column(Vector(1536), nullable=False)  # openai/text-embedding-3-small via OpenRouter
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to document
    document = relationship("Document", back_populates="chunks")


# =============================================================================
# Database Operations
# =============================================================================

def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables and create HNSW index."""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create HNSW index for fast similarity search
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """))
        conn.commit()
    
    print("✅ Database initialized successfully!")


def create_document(db, filename: str, file_type: str, file_path: str, file_size: int) -> Document:
    """Create a new document record."""
    doc = Document(
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        file_size=file_size
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def add_chunks(db, document_id: int, chunks: List[dict]):
    """
    Add chunks with embeddings to the database.
    
    Args:
        db: Database session
        document_id: ID of the parent document
        chunks: List of dicts with 'content' and 'embedding' keys
    """
    chunk_objects = []
    for idx, chunk in enumerate(chunks):
        chunk_obj = DocumentChunk(
            document_id=document_id,
            chunk_index=idx,
            content=chunk['content'],
            embedding=chunk['embedding']
        )
        chunk_objects.append(chunk_obj)
    
    db.add_all(chunk_objects)
    
    # Update document's total_chunks count
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc:
        doc.total_chunks = len(chunks)
    
    db.commit()


def search_similar_chunks(db, query_embedding: List[float], document_id: int, top_k: int = 5) -> List[DocumentChunk]:
    """
    Find the most similar chunks to the query embedding using cosine similarity.
    
    Args:
        db: Database session
        query_embedding: The embedding vector of the query
        document_id: ID of the document to search within
        top_k: Number of results to return
    
    Returns:
        List of DocumentChunk objects ordered by similarity
    """
    # Convert to numpy array for pgvector
    query_vector = np.array(query_embedding)
    
    # Query using cosine distance (1 - cosine_similarity)
    # Lower distance = higher similarity
    results = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).order_by(
        DocumentChunk.embedding.cosine_distance(query_vector)
    ).limit(top_k).all()
    
    return results


def get_document(db, document_id: int) -> Optional[Document]:
    """Get a document by ID."""
    return db.query(Document).filter(Document.id == document_id).first()


def get_all_documents(db) -> List[Document]:
    """Get all documents."""
    return db.query(Document).order_by(Document.created_at.desc()).all()


def delete_document(db, document_id: int) -> bool:
    """Delete a document and its chunks."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc:
        db.delete(doc)
        db.commit()
        return True
    return False
