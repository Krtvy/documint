"""
DocuMint Pydantic Models
Request and response models for the API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# =============================================================================
# Request Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for document queries."""
    document_id: int = Field(..., description="ID of the document to query")
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask about the document")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


# =============================================================================
# Response Models
# =============================================================================

class SourceChunk(BaseModel):
    """A source chunk used in the answer."""
    chunk_index: int
    preview: str


class QueryResponse(BaseModel):
    """Response model for document queries."""
    answer: str
    sources: List[SourceChunk]
    model: str
    chunks_used: int
    document_id: int
    question: str


class DocumentResponse(BaseModel):
    """Response model for document information."""
    id: int
    filename: str
    file_type: str
    file_size: int
    total_chunks: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentResponse]
    total: int


class UploadResponse(BaseModel):
    """Response model for document upload."""
    message: str
    document_id: int
    filename: str
    file_type: str
    total_chunks: int
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    embedding_model: str
