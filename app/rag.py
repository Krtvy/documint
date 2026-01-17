"""
DocuMint RAG Pipeline
Retrieval Augmented Generation using pgvector + Claude API.
"""

from typing import List, Dict
from anthropic import Anthropic

from app.config import get_settings
from app.database import search_similar_chunks
from app.embeddings import generate_embedding

settings = get_settings()


# =========================
# CLAUDE CLIENT
# =========================
def get_claude_client() -> Anthropic:
    """
    Create and return Anthropic Claude client.
    """
    return Anthropic(api_key=settings.anthropic_api_key)


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
    Generate answer using Claude based ONLY on retrieved context.
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

    # ---- Claude call
    client = get_claude_client()

    try:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        answer = response.content[0].text

    except Exception:
        answer = (
            "I encountered an error while generating the answer. "
            "Please try again."
        )

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
        "model": settings.claude_model,
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
            "model": settings.claude_model,
            "chunks_used": 0,
        }

    return generate_answer(
        query=query,
        context_chunks=context_chunks,
        document_name=document_name,
    )
