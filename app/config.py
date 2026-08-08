"""
DocuMint Configuration
Loads settings from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/documint"

    # OpenRouter. One key covers both embeddings and generation, and it speaks
    # the OpenAI wire format, so the standard SDK works against it unchanged.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # File Upload
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 10

    # Text Processing
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Models
    # Embeddings run over the API rather than a local model. sentence-transformers
    # pulls in PyTorch, which does not fit in a 512MB container.
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Free models on OpenRouter get rate limited without warning, so the first
    # choice is not always available. These are tried in order.
    generation_model: str = "google/gemma-4-26b-a4b-it:free"
    fallback_models: str = (
        "google/gemma-4-31b-it:free,"
        "nvidia/nemotron-3-super-120b-a12b:free,"
        "inclusionai/ling-3.0-tiny:free"
    )

    # Vector Search
    top_k_results: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create uploads directory if it doesn't exist
settings = get_settings()
os.makedirs(settings.upload_dir, exist_ok=True)