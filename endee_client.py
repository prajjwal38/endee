"""
endee_client.py
───────────────
Thin wrapper around the Endee Python SDK that handles:
  • Connecting to the locally running Endee Docker instance.
  • Creating the "arxiv_papers" index (idempotent – safe to call on every run).
  • Upserting document vectors with metadata.
  • Querying for the top-k nearest neighbours.

All configuration is read from environment variables (loaded via python-dotenv).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# ── SDK import ─────────────────────────────────────────────────────────────────
try:
    import endee  # Endee Python SDK  (pip install endee)
except ImportError as exc:
    raise ImportError(
        "Endee SDK not found. Install it with: pip install endee"
    ) from exc

# ── Configuration ──────────────────────────────────────────────────────────────
ENDEE_URL: str = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_AUTH_TOKEN: Optional[str] = os.getenv("ENDEE_AUTH_TOKEN") or None

INDEX_NAME: str = "arxiv_papers"
EMBEDDING_DIM: int = 768          # must match embeddings.py EMBEDDING_DIM (google/embeddinggemma-300m)
SPACE_TYPE: str = "cosine"        # cosine similarity for semantic search


# ── Client initialisation ──────────────────────────────────────────────────────

def _build_client() -> Any:
    """
    Instantiate the Endee client.
    Supports both authenticated and unauthenticated deployments.
    """
    print(f"[EndeeClient] Connecting to Endee at {ENDEE_URL} (parameter omitted in client instantiation) …")
    if ENDEE_AUTH_TOKEN:
        client = endee.Endee(token=ENDEE_AUTH_TOKEN)
    else:
        client = endee.Endee()
    return client


# Global singleton client
_client: Any = None


def get_client() -> Any:
    """Return (or lazily create) the global Endee client."""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# ── Index management ───────────────────────────────────────────────────────────

def ensure_index() -> None:
    """
    Create the 'arxiv_papers' index if it does not already exist.
    This is idempotent – calling it multiple times is safe.
    """
    client = get_client()
    try:
        existing_indexes = client.list_indexes()
        index_names = [idx.name if hasattr(idx, "name") else idx for idx in existing_indexes]

        if INDEX_NAME not in index_names:
            print(f"[EndeeClient] Creating index '{INDEX_NAME}' (dim={EMBEDDING_DIM}, space={SPACE_TYPE}) …")
            client.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                space_type=SPACE_TYPE,
            )
            print(f"[EndeeClient] Index '{INDEX_NAME}' created successfully.")
        else:
            print(f"[EndeeClient] Index '{INDEX_NAME}' already exists – skipping creation.")
    except Exception as exc:  # noqa: BLE001
        # Some SDK versions raise if the index already exists – treat as no-op
        if "already exists" in str(exc).lower():
            print(f"[EndeeClient] Index '{INDEX_NAME}' already exists (exception path) – continuing.")
        else:
            raise


# ── Upsert ─────────────────────────────────────────────────────────────────────

def upsert_vectors(records: List[Dict[str, Any]]) -> None:
    """
    Upsert a batch of vector records into the Endee index.

    Each record in `records` must have the following keys:
        id       (str)  – unique identifier for the chunk
        vector   (List[float]) – embedding vector
        metadata (dict) – arbitrary key/value pairs stored alongside the vector
                          e.g. {"title": "…", "chunk_index": 0, "source_url": "…"}

    Parameters
    ----------
    records : List[Dict[str, Any]]
        Batch of records to upsert.
    """
    client = get_client()
    print(f"[EndeeClient] Upserting {len(records)} vector(s) into '{INDEX_NAME}' …")

    # Endee SDK upsert signature may vary by version; we handle both styles.
    try:
        idx = client.get_index(INDEX_NAME)
        idx.upsert(vectors=records)
    except TypeError:
        # Fallback: some versions use positional args or different kwarg names
        idx.upsert(records)

    print(f"[EndeeClient] Upsert complete.")


# ── Query ──────────────────────────────────────────────────────────────────────

def query_index(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the Endee index for the top-k nearest neighbours to `query_vector`.

    Parameters
    ----------
    query_vector : List[float]
        Embedded representation of the user's query.
    top_k : int
        Number of results to return (default 5).

    Returns
    -------
    List[Dict[str, Any]]
        Each item contains at least:
            id       (str)
            score    (float) – similarity score (higher = more similar for cosine)
            metadata (dict)  – stored metadata for the chunk
    """
    client = get_client()
    print(f"[EndeeClient] Querying '{INDEX_NAME}' for top-{top_k} matches …")

    try:
        idx = client.get_index(INDEX_NAME)
        results = idx.query(
            vector=query_vector,
            top_k=top_k,
            ef=max(top_k * 10, 128),   # wider HNSW beam = more candidates returned
        )
    except TypeError:
        results = idx.query(query_vector, top_k)

    # Normalise results into a consistent dict format regardless of SDK version
    # Endee returns: id, similarity, distance, meta, norm, filter, vector
    normalised: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, dict):
            normalised.append({
                "id": r.get("id", ""),
                "score": r.get("similarity", r.get("score", 0.0)),
                "metadata": r.get("meta", r.get("metadata", {})),
            })
        else:
            # Object with attributes
            normalised.append({
                "id": getattr(r, "id", ""),
                "score": getattr(r, "similarity", getattr(r, "score", 0.0)),
                "metadata": getattr(r, "meta", getattr(r, "metadata", {})),
            })

    print(f"[EndeeClient] Retrieved {len(normalised)} result(s).")
    return normalised
