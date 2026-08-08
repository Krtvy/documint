"""
DocuMint Embeddings

Embeddings are generated over the OpenRouter API rather than a local model.

Why: this used to run sentence-transformers with all-MiniLM-L6-v2, which pulls
PyTorch in as a dependency. That is roughly 2GB installed and needs more RAM
than a small container has, so the service could not be deployed on a 512MB
instance. Moving to an API call removes the heavy dependency entirely. The
tradeoff is a network round trip per batch, which is why batching matters here.

OpenRouter speaks the OpenAI wire format, so the openai SDK works against it
with only the base_url changed.
"""

from typing import List

import numpy as np
from openai import OpenAI

from app.config import get_settings

settings = get_settings()

_client = None

# Keep batches modest so one failed request does not cost a whole document.
_MAX_BATCH = 64


def get_client() -> OpenAI:
    """Return a cached client pointed at OpenRouter."""
    global _client
    if _client is None:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Embeddings cannot be generated."
            )
        _client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
    return _client


def _embed(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts, batching to stay within request limits."""
    client = get_client()

    out: List[List[float]] = []
    for start in range(0, len(texts), _MAX_BATCH):
        batch = texts[start : start + _MAX_BATCH]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        # The API does not promise ordering, so sort by index before using.
        ordered = sorted(response.data, key=lambda d: d.index)
        out.extend(d.embedding for d in ordered)

    return out


def generate_embedding(text: str) -> List[float]:
    """Embed a single search query."""
    return _embed([text])[0]


def generate_embeddings(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Embed many document chunks at once.

    batch_size is accepted for backwards compatibility with existing callers,
    but the internal batch limit is what actually governs request size.
    """
    if not texts:
        return []
    return _embed(texts)


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Cosine similarity between two vectors, for comparing without a database hit.

    Returns 0.0 when either vector has no magnitude, since the angle between
    them is undefined in that case.
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def preload_model():
    """
    Kept so startup code that calls this still works.

    There is no local model to warm up now. This builds the client early so a
    missing API key fails at boot instead of on the first upload.
    """
    get_client()
