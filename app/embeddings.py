"""
DocuMint Embeddings
Generates vector embeddings using sentence-transformers.
Uses the all-MiniLM-L6-v2 model (384 dimensions, fast and good quality).
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

from app.config import get_settings

settings = get_settings()

# Global model instance (loaded once, reused)
_model = None


def get_model() -> SentenceTransformer:
    """
    Get or load the embedding model.
    
    Uses a global instance to avoid reloading the model
    on every request (models are large and slow to load).
    """
    global _model
    
    if _model is None:
        print(f"Loading embedding model: {settings.embedding_model}...")
        _model = SentenceTransformer(settings.embedding_model)
        print("✅ Embedding model loaded!")
    
    return _model


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text string.
    
    Args:
        text: The text to embed
    
    Returns:
        List of floats representing the embedding vector (384 dimensions)
    """
    model = get_model()
    
    # Generate embedding
    embedding = model.encode(text, convert_to_numpy=True)
    
    # Convert to list for JSON serialization and database storage
    return embedding.tolist()


def generate_embeddings(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for multiple texts (batch processing).
    
    More efficient than calling generate_embedding() in a loop.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process at once
    
    Returns:
        List of embedding vectors
    """
    model = get_model()
    
    # Generate embeddings in batch
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 100  # Show progress for large batches
    )
    
    # Convert to list of lists
    return embeddings.tolist()


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Useful for comparing similarity without database lookup.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Similarity score between -1 and 1 (1 = identical)
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Cosine similarity formula: dot(a,b) / (norm(a) * norm(b))
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def preload_model():
    """
    Preload the model at application startup.
    
    Call this during FastAPI startup to avoid
    loading delay on first request.
    """
    get_model()
