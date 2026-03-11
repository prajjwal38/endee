"""
embeddings.py
─────────────
Centralised embedding helper used by both the Ingestion Agent (to embed
text chunks) and the RAG Query Agent (to embed the user query).

Model : google/embeddinggemma-300m  (EmbeddingGemma by Google)
  • Based on Gemma 3 / T5Gemma, released on HuggingFace
  • Embedding dimension : 768 (default; supports MRL truncation to 512/256/128)
  • Supports 100+ languages
  • Runs via sentence-transformers library
  • ⚠️  Does NOT support float16 – use float32 or bfloat16

Prompting convention for EmbeddingGemma:
  • Queries   → prefix: "task: {task} | query: {text}"
  • Documents → prefix: "title: {title or 'none'} | text: {text}"
"""

from __future__ import annotations

from typing import List
from sentence_transformers import SentenceTransformer

# ── Model configuration ────────────────────────────────────────────────────────
MODEL_NAME: str = "google/embeddinggemma-300m"
EMBEDDING_DIM: int = 768          # Default output dimensionality

# Task description injected into query prompts for EmbeddingGemma
QUERY_TASK: str = "Given a scientific research question, retrieve relevant paper passages"

# Singleton – loaded once per process to avoid repeated disk I/O
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the EmbeddingGemma model (loaded once per process)."""
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model '{MODEL_NAME}' — first run may take a moment …")
        # EmbeddingGemma must use float32 (not float16)
        _model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": "float32"})
        print(f"[Embeddings] Model loaded. Output dim = {EMBEDDING_DIM}")
    return _model


# ── Query embedding ────────────────────────────────────────────────────────────

def embed_query(query: str) -> List[float]:
    """
    Embed a *user query* using EmbeddingGemma's recommended query prompt format.

    Parameters
    ----------
    query : str  –  The raw user question.

    Returns
    -------
    List[float]  –  Dense vector of length EMBEDDING_DIM (768).
    """
    model = _get_model()
    # EmbeddingGemma query prompt format
    prompted = f"task: {QUERY_TASK} | query: {query}"
    vector = model.encode(prompted, convert_to_numpy=True)
    return vector.tolist()


# ── Document embedding ─────────────────────────────────────────────────────────

def embed_text(text: str, title: str = "none") -> List[float]:
    """
    Embed a *document chunk* using EmbeddingGemma's recommended document prompt format.

    Parameters
    ----------
    text  : str  –  The chunk of text to embed.
    title : str  –  Paper title (or "none" if not available).

    Returns
    -------
    List[float]  –  Dense vector of length EMBEDDING_DIM (768).
    """
    model = _get_model()
    prompted = f"title: {title} | text: {text}"
    vector = model.encode(prompted, convert_to_numpy=True)
    return vector.tolist()


def embed_batch(
    texts: List[str],
    titles: List[str] | None = None,
) -> List[List[float]]:
    """
    Embed a batch of document chunks in one forward pass (more efficient).

    Parameters
    ----------
    texts  : List[str]  –  Chunks to embed.
    titles : List[str]  –  Corresponding paper titles (optional).

    Returns
    -------
    List[List[float]]  –  One vector per input chunk.
    """
    model = _get_model()
    if titles is None:
        titles = ["none"] * len(texts)

    prompted = [
        f"title: {t} | text: {c}"
        for t, c in zip(titles, texts)
    ]
    vectors = model.encode(prompted, convert_to_numpy=True, batch_size=16)
    return [v.tolist() for v in vectors]
