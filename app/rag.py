"""
DocuMint RAG Pipeline
Retrieval Augmented Generation using pgvector + OpenRouter.
"""

from typing import List, Dict

from openai import OpenAI

from app.config import get_settings
from app.database import search_similar_chunks
from app.embeddings import generate_embedding

settings = get_settings()

_client = None


# =========================
# CLIENT
# =========================
def get_client() -> OpenAI:
    """
    Return a cached client pointed at OpenRouter.
    """
    global _client
    if _client is None:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Cannot generate answers."
            )
        _client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
    return _client


# =========================
# RETRIEVAL
# =========================
def retrieve_context(
    db,
    document_id: int,
    query: str,
    top_k: int | None = None,
) -> List[Dict]:
    """
    Retrieve relevant document chunks using vector similarity search.
    """
    top_k = top_k or settings.top_k_results

    # Generate embedding for user query
    query_embedding = generate_embedding(query)

    # Vector similarity search (pgvector)
    chunks = search_similar_chunks(
        db=db,
        query_embedding=query_embedding,
        document_id=document_id,
        top_k=top_k,
    )

    # Format chunks
    return [
        {
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
        }
        for chunk in chunks
    ]


# =========================
# GENERATION
# =========================
def generate_answer(
    query: str,
    context_chunks: List[Dict],
    document_name: str,
) -> Dict:
    """
    Generate an answer from the retrieved context only.
    """

    # ---- Build context safely (prevent prompt explosion)
    MAX_CHARS_PER_CHUNK = 1500

    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        content = chunk["content"][:MAX_CHARS_PER_CHUNK]
        context_parts.append(
            f"[Source {i}] (Chunk {chunk['chunk_index'] + 1}):\n{content}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    # ---- Prompts
    system_prompt = (
        "You are a helpful document assistant.\n"
        "Answer ONLY using the provided document context.\n\n"
        "Rules:\n"
        "1. Do NOT use outside knowledge\n"
        "2. If information is missing, say so clearly\n"
        "3. Cite sources using [Source N]\n"
        "4. Be concise and factual\n"
    )

    user_prompt = f"""
Document: {document_name}

Context:
{context_str}

Question:
{query}

Answer using only the context above.
"""

    # ---- Generation call
    client = get_client()

    # Free models get rate limited without warning, so walk the fallback list
    # rather than failing the whole request on the first 429.
    candidates = [settings.generation_model] + [
        m.strip() for m in settings.fallback_models.split(",") if m.strip()
    ]

    answer = None
    model_used = None
    last_error = None

    for model in candidates:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                answer, model_used = text, model
                break
            last_error = "empty response"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    if answer is None:
        # Surface the reason rather than hiding it. A silent generic failure
        # here is what makes RAG systems impossible to debug in production.
        answer = f"Could not generate an answer. Last error: {last_error}"
        model_used = "none"

    # ---- Source metadata
    sources = [
        {
            "chunk_index": chunk["chunk_index"],
            "preview": (
                chunk["content"][:200] + "..."
                if len(chunk["content"]) > 200
                else chunk["content"]
            ),
        }
        for chunk in context_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "model": model_used,
        "chunks_used": len(context_chunks),
    }


# =========================
# FULL RAG PIPELINE
# =========================
def query_document(
    db,
    document_id: int,
    query: str,
    document_name: str,
    top_k: int | None = None,
) -> Dict:
    """
    End-to-end RAG:
    Retrieval → Context → Generation
    """

    context_chunks = retrieve_context(
        db=db,
        document_id=document_id,
        query=query,
        top_k=top_k,
    )

    if not context_chunks:
        return {
            "answer": (
                "I couldn't find relevant information in the document "
                "to answer your question."
            ),
            "sources": [],
            "model": settings.generation_model,
            "chunks_used": 0,
        }

    return generate_answer(
        query=query,
        context_chunks=context_chunks,
        document_name=document_name,
    )
