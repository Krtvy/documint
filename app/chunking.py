"""
DocuMint Text Chunking
Splits extracted text into manageable chunks for embedding.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from app.config import get_settings

settings = get_settings()


def create_chunks(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Uses RecursiveCharacterTextSplitter which tries to split on:
    1. Paragraphs (double newlines)
    2. Sentences (periods, question marks, etc.)
    3. Words (spaces)
    4. Characters (as last resort)
    
    Args:
        text: The full document text to split
        chunk_size: Maximum characters per chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)
    
    Returns:
        List of text chunks
    """
    # Use defaults from config if not specified
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    # Create the text splitter
    # RecursiveCharacterTextSplitter is the recommended splitter for most use cases
    # It tries to keep semantically related text together
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "? ",    # Questions
            "! ",    # Exclamations
            "; ",    # Semi-colons
            ", ",    # Commas
            " ",     # Words
            ""       # Characters (last resort)
        ]
    )
    
    # Split the text
    chunks = splitter.split_text(text)
    
    # Filter out empty or whitespace-only chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    return chunks


def create_chunks_with_metadata(text: str, source: str = None) -> List[dict]:
    """
    Split text into chunks with metadata.
    
    Args:
        text: The full document text
        source: Source identifier (e.g., filename)
    
    Returns:
        List of dicts with 'content' and 'metadata' keys
    """
    chunks = create_chunks(text)
    
    return [
        {
            'content': chunk,
            'metadata': {
                'source': source,
                'chunk_index': idx,
                'total_chunks': len(chunks)
            }
        }
        for idx, chunk in enumerate(chunks)
    ]


def estimate_chunks(text: str, chunk_size: int = None) -> int:
    """
    Estimate how many chunks will be created from text.
    Useful for progress indicators.
    
    Args:
        text: The text to estimate
        chunk_size: Chunk size to use for estimation
    
    Returns:
        Estimated number of chunks
    """
    chunk_size = chunk_size or settings.chunk_size
    
    if not text:
        return 0
    
    # Simple estimation: text length / (chunk_size - overlap)
    effective_chunk_size = chunk_size - settings.chunk_overlap
    return max(1, len(text) // effective_chunk_size + 1)
