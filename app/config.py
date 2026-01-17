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
    
    # Anthropic API
    anthropic_api_key: str = ""
    
    # File Upload
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 10
    
    # Text Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    claude_model: str = "claude-sonnet-4-20250514"
    
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