"""
agents/ingestion_agent.py
──────────────────────────
Ingestion Agent – downloads, chunks, embeds, and stores ArXiv papers.

Pipeline for each paper:
  1. Download full text from PDF URL using PyMuPDF (fitz).
     If PDF download fails, fall back to the paper's abstract.
  2. Split the text into overlapping ~500-token chunks (≈ 400 words).
  3. Embed each chunk using google/embeddinggemma-300m (via embeddings.py).
  4. Upsert vectors + metadata into the Endee "arxiv_papers" index.
"""

from __future__ import annotations

import hashlib
import io
import textwrap
from typing import Any, Dict, List

import requests
from crewai import Agent, Task

from embeddings import embed_batch
from endee_client import ensure_index, upsert_vectors

# ── Chunking configuration ─────────────────────────────────────────────────────
CHUNK_SIZE_WORDS: int = 400    # ~500 tokens  (1 token ≈ 0.75 words)
CHUNK_OVERLAP_WORDS: int = 40  # ~50 token overlap

# ── PDF parsing ────────────────────────────────────────────────────────────────

def _download_pdf_text(pdf_url: str) -> str:
    """
    Download a PDF from `pdf_url` and extract its plain text using PyMuPDF.
    Returns an empty string on any failure.
    """
    try:
        import fitz  # PyMuPDF

        print(f"  [Ingestion] Downloading PDF: {pdf_url} …")
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()

        doc = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
        pages = [page.get_text() for page in doc]
        full_text = "\n".join(pages)
        print(f"  [Ingestion] PDF extracted: {len(full_text)} characters.")
        return full_text

    except Exception as exc:  # noqa: BLE001
        print(f"  [Ingestion] PDF download/parse failed ({exc}). Using abstract instead.")
        return ""


# ── Text chunking ──────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """
    Split `text` into overlapping word-based chunks.

    Parameters
    ----------
    text       : str – Full text to chunk.
    chunk_size : int – Number of words per chunk.
    overlap    : int – Number of words shared between consecutive chunks.

    Returns
    -------
    List[str] – Non-empty text chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start: start + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break

    return chunks


# ── Core ingestion logic ───────────────────────────────────────────────────────

def ingest_papers(papers: List[Dict[str, Any]]) -> int:
    """
    Chunk, embed, and upsert a list of ArXiv papers into Endee.

    Parameters
    ----------
    papers : List[Dict] – Output from web_search_agent.fetch_arxiv_papers().

    Returns
    -------
    int – Total number of vector chunks upserted.
    """
    ensure_index()
    total_upserted = 0

    for paper in papers:
        title = paper.get("title", "Unknown")
        pdf_url = paper.get("pdf_url", "")
        entry_url = paper.get("entry_url", "")
        abstract = paper.get("abstract", "")

        print(f"\n[IngestionAgent] Processing: {title[:70]} …")

        # 1. Get full text (PDF preferred, abstract fallback)
        full_text = _download_pdf_text(pdf_url) if pdf_url else ""
        if len(full_text.strip()) < 200:
            print("  [Ingestion] Using abstract as text source.")
            full_text = abstract

        # 2. Chunk the text
        chunks = _chunk_text(full_text)
        if not chunks:
            print("  [Ingestion] No text to chunk — skipping paper.")
            continue

        print(f"  [Ingestion] {len(chunks)} chunk(s) generated.")

        # 3. Embed all chunks in one batched call
        titles_repeated = [title] * len(chunks)
        vectors = embed_batch(chunks, titles=titles_repeated)

        # 4. Build Endee records
        records = []
        for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            # Deterministic ID: hash of (arxiv_id + chunk_index)
            arxiv_id = paper.get("arxiv_id", hashlib.md5(entry_url.encode()).hexdigest()[:8])
            record_id = f"{arxiv_id}_chunk_{idx}"

            records.append({
                "id": record_id,
                "vector": vector,
                "meta": {                       # Endee VectorItem uses 'meta' not 'metadata'
                    "title": title,
                    "chunk_index": idx,
                    "source_url": entry_url,
                    "pdf_url": pdf_url,
                    "text": chunk_text[:1000],   # store first 1000 chars for retrieval
                },
            })

        # 5. Upsert into Endee
        upsert_vectors(records)
        total_upserted += len(records)

    print(f"\n[IngestionAgent] Ingestion complete. Total chunks upserted: {total_upserted}")
    return total_upserted


# ── CrewAI Agent & Task factory ────────────────────────────────────────────────

def build_ingestion_agent() -> Agent:
    """Create and return the CrewAI Ingestion Agent."""
    return Agent(
        role="Research Paper Ingestion Specialist",
        goal=(
            "Download, parse, chunk, embed, and store ArXiv paper content "
            "into the Endee vector database for later retrieval."
        ),
        backstory=(
            "You are a meticulous data engineer specialising in scientific "
            "knowledge graph construction. You transform raw research papers "
            "into searchable vector embeddings stored in Endee."
        ),
        verbose=True,
        allow_delegation=False,
    )


def build_ingestion_task(agent: Agent, papers: List[Dict[str, Any]]) -> Task:
    """
    Build the CrewAI Task for the Ingestion Agent.

    Parameters
    ----------
    agent  : Agent          – The ingestion agent instance.
    papers : List[Dict]     – Papers from the Web Search Agent.
    """
    total = ingest_papers(papers)

    return Task(
        description=textwrap.dedent(f"""
            You have received {len(papers)} ArXiv papers from the Web Search Agent.
            The Ingestion pipeline has already:
              1. Downloaded and parsed each paper's PDF (with abstract fallback).
              2. Split the text into overlapping ~500-token chunks.
              3. Embedded every chunk using google/embeddinggemma-300m (768-dim).
              4. Upserted {total} vector chunks into the Endee 'arxiv_papers' index.

            Your task: Confirm the ingestion was successful and report the total
            number of chunks stored. Pass this information to the RAG Query Agent.
        """),
        agent=agent,
        expected_output=f"Confirmation that {total} chunks from {len(papers)} papers were successfully upserted into Endee.",
    )
