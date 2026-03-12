"""
agents/ingestion_agent.py
──────────────────────────
Ingestion Agent – downloads, chunks, embeds, and stores ArXiv papers.

Pipeline for each paper:
  1. Download the LaTeX source tarball from ArXiv (https://arxiv.org/e-print/<id>).
     Falls back to PDF (PyMuPDF) if tarball extraction fails, and then to the
     abstract if PDF also fails.
  2. Extract & clean the primary .tex file from the tarball (preserving LaTeX math).
  3. Send the cleaned LaTeX to a local Ollama model (gpt-oss:120b, turbo mode) to
     semantically chunk the document by logical section/topic boundaries.
  4. Embed each semantic chunk using google/embeddinggemma-300m (via embeddings.py).
  5. Upsert vectors + enriched metadata into the Endee "arxiv_papers" index.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tarfile
import textwrap
from typing import Any, Dict, List, Optional

import requests
from crewai import Agent, Task
from dotenv import load_dotenv

from embeddings import embed_batch
from endee_client import ensure_index, upsert_vectors

load_dotenv()

# ── Ollama configuration ───────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL",    "gpt-oss:120b")

# Maximum characters to send to Ollama per request
# gpt-oss:120b in turbo mode handles very large contexts, but we cap to be safe
OLLAMA_MAX_CHARS: int = 80_000

# Fallback: character-based chunk size used only when Ollama is unavailable
FALLBACK_CHUNK_CHARS: int = 3000
FALLBACK_OVERLAP_CHARS: int = 200


# ── LaTeX source extraction ────────────────────────────────────────────────────

def _download_latex_source(tarball_url: str) -> Optional[str]:
    """
    Download the ArXiv e-print tarball and extract the primary .tex file.

    Returns the cleaned LaTeX text, or None on failure.
    """
    try:
        print(f"  [Ingestion] Downloading LaTeX tarball: {tarball_url} …")
        resp = requests.get(tarball_url, timeout=60, stream=True)
        resp.raise_for_status()

        content = resp.content

        # ArXiv may serve a plain .tex or a .tar.gz depending on the paper
        # Try as tarball first
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                tex_files = [m for m in tar.getmembers() if m.name.endswith(".tex")]

                if not tex_files:
                    print("  [Ingestion] No .tex files found inside tarball.")
                    return None

                # Pick the largest .tex file (usually the main document)
                main_tex = max(tex_files, key=lambda m: m.size)
                print(f"  [Ingestion] Extracting: {main_tex.name} ({main_tex.size:,} bytes)")
                f = tar.extractfile(main_tex)
                if f is None:
                    return None
                raw_tex = f.read().decode("utf-8", errors="replace")

        except tarfile.TarError:
            # Might be a raw .tex file (some old papers)
            raw_tex = content.decode("utf-8", errors="replace")
            if not raw_tex.strip().startswith("\\") and "\\document" not in raw_tex[:500]:
                print("  [Ingestion] Downloaded content doesn't look like LaTeX.")
                return None

        cleaned = _clean_latex(raw_tex)
        print(f"  [Ingestion] LaTeX extracted: {len(cleaned):,} characters.")
        return cleaned

    except Exception as exc:  # noqa: BLE001
        print(f"  [Ingestion] LaTeX tarball failed ({exc}).")
        return None


def _clean_latex(tex: str) -> str:
    """
    Lightly clean a LaTeX source file for LLM consumption.

    Keeps:  \\section, \\subsection, \\begin{equation}, equation content,
            paragraph text, \\cite, \\ref.
    Removes: preamble (before \\begin{document}), \\usepackage, comments (%),
             \\newcommand, figure/table environments (too noisy without images).
    """
    # Strip everything before \begin{document}
    doc_match = re.search(r"\\begin\{document\}", tex)
    if doc_match:
        tex = tex[doc_match.start():]

    # Remove LaTeX comments
    tex = re.sub(r"%[^\n]*", "", tex)

    # Remove figure and table environments (they contain captions/labels but no real text)
    tex = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "", tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "", tex, flags=re.DOTALL)

    # Remove bibliography entries
    tex = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", tex, flags=re.DOTALL)

    # Collapse excessive whitespace / blank lines
    tex = re.sub(r"\n{3,}", "\n\n", tex)

    return tex.strip()


# ── PDF fallback extraction ────────────────────────────────────────────────────

def _download_pdf_text(pdf_url: str) -> str:
    """Fallback: extract flat text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        print(f"  [Ingestion] Falling back to PDF: {pdf_url} …")
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
        doc = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
        pages = [page.get_text() for page in doc]
        full_text = "\n".join(pages)
        print(f"  [Ingestion] PDF extracted: {len(full_text):,} characters.")
        return full_text

    except Exception as exc:
        print(f"  [Ingestion] PDF fallback failed ({exc}).")
        return ""


# ── Ollama LLM Chunking ────────────────────────────────────────────────────────

def _chunk_with_ollama(text: str, paper_title: str) -> List[str]:
    """
    Send the document text to the local Ollama model (gpt-oss:120b, turbo mode)
    for semantic chunking.

    The model is instructed to split the document by logical section/topic
    boundaries, preserving LaTeX equation blocks intact.

    Returns a list of chunk strings. Falls back to character-based chunking
    if Ollama is unavailable or returns an unusable response.
    """
    # Trim to max allowed characters to stay within context window
    text_to_send = text[:OLLAMA_MAX_CHARS]

    prompt = (
        f"You are processing an academic physics paper titled: \"{paper_title}\".\n\n"
        "Your task is to split the following LaTeX/text document into semantically "
        "coherent chunks. Rules:\n"
        "1. Each chunk should represent ONE complete idea, theorem, derivation, "
        "   or logical section. Do NOT cut mid-sentence or mid-equation.\n"
        "2. Preserve ALL LaTeX equation blocks (\\begin{equation}...\\end{equation}, "
        "   inline $...$, display $$...$$) entirely within a single chunk.\n"
        "3. Aim for chunks of 300-800 words. A chunk may be shorter if it covers "
        "   a complete discrete idea (e.g., a single equation + its explanation).\n"
        "4. Separate chunks with the exact delimiter: ---CHUNK---\n\n"
        "Document:\n"
        "---\n"
        f"{text_to_send}\n"
        "---\n\n"
        "Output ONLY the chunked text with ---CHUNK--- delimiters. "
        "Do not add commentary."
    )

    try:
        print(f"  [Ingestion] Sending {len(text_to_send):,} chars to Ollama ({OLLAMA_MODEL}) for chunking …")
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 16000,   # allow generous output
                    "temperature": 0.0,     # deterministic
                },
            },
            timeout=300,  # large model may take time
        )
        resp.raise_for_status()
        ollama_output = resp.json().get("response", "")

        chunks = [c.strip() for c in ollama_output.split("---CHUNK---") if c.strip()]
        if len(chunks) < 2:
            print("  [Ingestion] Ollama returned <2 chunks — falling back to character chunking.")
            return _chunk_by_chars(text)

        print(f"  [Ingestion] Ollama produced {len(chunks)} semantic chunk(s).")
        return chunks

    except Exception as exc:
        print(f"  [Ingestion] Ollama unavailable ({exc}). Falling back to character chunking.")
        return _chunk_by_chars(text)


def _chunk_by_chars(text: str, size: int = FALLBACK_CHUNK_CHARS, overlap: int = FALLBACK_OVERLAP_CHARS) -> List[str]:
    """
    Simple character-based sliding window. Used as a fallback when Ollama
    is unavailable. Tries to break at the nearest sentence boundary.
    """
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # Try to break at the last sentence boundary within the window
            boundary = text.rfind(". ", start, end)
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


# ── Section heading extraction ─────────────────────────────────────────────────

def _extract_section_heading(chunk_text: str) -> str:
    """
    Attempt to extract the enclosing section/subsection heading from a chunk.
    Looks for \\section{...} or \\subsection{...} at or near the top of the chunk.
    Returns the heading string or 'Body' if none found.
    """
    match = re.search(r"\\(?:sub)*section\*?\{([^}]+)\}", chunk_text)
    if match:
        return match.group(1).strip()
    return "Body"


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
        title            = paper.get("title", "Unknown")
        arxiv_id         = paper.get("arxiv_id", "")
        entry_url        = paper.get("entry_url", "")
        pdf_url          = paper.get("pdf_url", "")
        abstract         = paper.get("abstract", "")
        tarball_url      = paper.get("source_tarball_url", f"https://arxiv.org/e-print/{arxiv_id}")
        published        = paper.get("published", "")
        updated          = paper.get("updated", "")
        primary_category = paper.get("primary_category", "")
        categories       = paper.get("categories", [])
        authors          = paper.get("authors", [])

        print(f"\n[IngestionAgent] Processing: {title[:70]} …")

        # ── 1. Acquire document text ───────────────────────────────────────────
        # Priority: LaTeX tarball  →  PDF fallback  →  abstract
        full_text = _download_latex_source(tarball_url)
        source_type = "latex"

        if not full_text or len(full_text.strip()) < 200:
            pdf_text = _download_pdf_text(pdf_url) if pdf_url else ""
            if len(pdf_text.strip()) >= 200:
                full_text = pdf_text
                source_type = "pdf"
            else:
                print("  [Ingestion] Using abstract as text source.")
                full_text = abstract
                source_type = "abstract"

        print(f"  [Ingestion] Source type: {source_type}")

        # ── 2. Chunk the text ──────────────────────────────────────────────────
        chunks = _chunk_with_ollama(full_text, title)
        if not chunks:
            print("  [Ingestion] No chunks produced — skipping paper.")
            continue

        print(f"  [Ingestion] {len(chunks)} chunk(s) ready for embedding.")

        # ── 3. Embed all chunks in one batched call ────────────────────────────
        titles_repeated = [title] * len(chunks)
        vectors = embed_batch(chunks, titles=titles_repeated)

        # ── 4. Build Endee records with enriched metadata ─────────────────────
        records = []
        for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            record_id = f"{arxiv_id}_chunk_{idx}"
            sub_heading = _extract_section_heading(chunk_text)

            records.append({
                "id":     record_id,
                "vector": vector,
                "meta": {
                    # ── Core identifiers ───────────────────────────────────────
                    "title":            title,
                    "chunk_index":      idx,
                    "source_url":       entry_url,
                    "pdf_url":          pdf_url,
                    "text":             chunk_text[:2000],  # store up to 2000 chars
                    "source_type":      source_type,        # "latex" | "pdf" | "abstract"
                    # ── Enriched metadata for filtering & hybrid search ────────
                    "published":        published,          # "YYYY-MM-DD"
                    "updated":          updated,            # "YYYY-MM-DD"
                    "primary_category": primary_category,   # e.g. "gr-qc"
                    "categories":       categories,         # e.g. ["gr-qc", "hep-th"]
                    "authors":          authors[:5],         # first 5 authors
                    "sub_heading":      sub_heading,         # section heading from LaTeX
                },
            })

        # ── 5. Upsert into Endee ───────────────────────────────────────────────
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
            "Download LaTeX source from ArXiv, semantically chunk via Ollama, "
            "embed, and store research paper content into the Endee vector database."
        ),
        backstory=(
            "You are a meticulous data engineer specialising in scientific "
            "knowledge graph construction. You transform raw LaTeX research papers "
            "into searchable vector embeddings stored in Endee, preserving "
            "mathematical equations and logical structure."
        ),
        verbose=True,
        allow_delegation=False,
    )


def build_ingestion_task(agent: Agent, papers: List[Dict[str, Any]]) -> Task:
    """Build the CrewAI Task for the Ingestion Agent."""
    total = ingest_papers(papers)

    return Task(
        description=textwrap.dedent(f"""
            You have received {len(papers)} ArXiv papers from the Web Search Agent.
            The Ingestion pipeline has already:
              1. Downloaded LaTeX source tarballs (with PDF/abstract fallback).
              2. Semantically chunked each document using Ollama ({OLLAMA_MODEL}).
              3. Embedded every chunk using google/embeddinggemma-300m (768-dim).
              4. Upserted {total} vector chunks with enriched metadata into Endee.

            Your task: Confirm the ingestion was successful and report the total
            number of chunks stored. Pass this information to the RAG Query Agent.
        """),
        agent=agent,
        expected_output=f"Confirmation that {total} chunks from {len(papers)} papers were successfully upserted into Endee.",
    )
