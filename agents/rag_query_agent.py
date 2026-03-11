"""
agents/rag_query_agent.py
──────────────────────────
RAG Query Agent – retrieves relevant context from Endee, re-ranks it
with Qwen3-Reranker-0.6B, and generates a grounded, cited answer using
Google Gemini.

Pipeline:
  1. Embed the user's query using google/embeddinggemma-300m (query prompt format).
  2. Query Endee "arxiv_papers" index for top-20 nearest candidate chunks (ANN).
  3. Re-rank the 20 candidates with Qwen3-Reranker-0.6B (cross-encoder).
  4. Take the top-5 re-ranked chunks and assemble a context block.
  5. Send context + user query to Gemini (gemini-1.5-flash) with a
     system instruction to answer ONLY from the provided context.
  6. Return the final answer with in-text citations (paper titles + URLs).
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List

import google.generativeai as genai
from crewai import Agent, Task
from dotenv import load_dotenv

from embeddings import embed_query
from endee_client import query_index
from reranker import rerank

# Load environment variables
load_dotenv()

# ── Gemini configuration ───────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.5-flash"   # fast and capable; swap to gemini-2.5-pro for higher quality

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Add it to your .env file or set it as an environment variable."
    )

genai.configure(api_key=GEMINI_API_KEY)


# ── Core RAG logic ─────────────────────────────────────────────────────────────

def rag_answer(user_query: str, candidate_k: int = 20, top_k: int = 5) -> str:
    """
    Full RAG pipeline: embed query → retrieve from Endee → rerank → generate with Gemini.

    Parameters
    ----------
    user_query  : str – The original research question from the user.
    candidate_k : int – Number of ANN candidates to pull from Endee before reranking (default 20).
    top_k       : int – Number of top chunks to keep after reranking for context (default 5).

    Returns
    -------
    str – Grounded, cited answer from Gemini.
    """
    # ── Step 1: Embed the query ────────────────────────────────────────────────
    print(f"\n[RAGQueryAgent] Embedding query …")
    query_vector = embed_query(user_query)

    # ── Step 2: Retrieve candidates from Endee (ANN, larger pool) ─────────────
    print(f"[RAGQueryAgent] Retrieving top-{candidate_k} ANN candidates from Endee …")
    candidates = query_index(query_vector, top_k=candidate_k)

    if not candidates:
        return (
            "I could not find any relevant passages in the knowledge base. "
            "Please try ingesting more papers first."
        )

    # ── Step 3: Re-rank with Qwen3-Reranker-0.6B ──────────────────────────────
    print(f"[RAGQueryAgent] Re-ranking {len(candidates)} candidate(s) with Qwen3-Reranker-0.6B …")
    results = rerank(user_query, candidates, top_k=top_k)

    # ── Step 4: Build context block ────────────────────────────────────────────
    context_parts: List[str] = []
    seen_sources: Dict[str, str] = {}   # title → source_url

    for rank, result in enumerate(results, start=1):
        meta = result.get("metadata", {})
        chunk_text = meta.get("text", "")
        title = meta.get("title", "Unknown Paper")
        source_url = meta.get("source_url", "")
        score = result.get("score", 0.0)

        seen_sources[title] = source_url

        context_parts.append(
            f"[Source {rank}] {title} (similarity: {score:.4f})\n"
            f"URL: {source_url}\n"
            f"{chunk_text}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    # ── Step 5: Call Gemini ────────────────────────────────────────────────────
    system_instruction = textwrap.dedent("""
        You are a scientific research assistant. Answer the user's question
        using ONLY the provided research paper excerpts (context below).
        Do not use any outside knowledge. If the context does not contain
        enough information to answer, say so explicitly.

        For every claim you make, cite the source using [Source N] notation.
        At the end of your answer, include a "References" section listing
        all cited papers with their titles and URLs.
    """).strip()

    prompt = textwrap.dedent(f"""
        CONTEXT FROM RESEARCH PAPERS:
        ─────────────────────────────
        {context_block}

        ─────────────────────────────
        USER QUESTION:
        {user_query}

        ─────────────────────────────
        Please provide a detailed, well-structured answer based solely on
        the context above. Cite sources using [Source N] notation inline.
        End with a References section.
    """).strip()

    print(f"[RAGQueryAgent] Sending context + query to Gemini ({GEMINI_MODEL}) …")
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_instruction,
    )
    response = model.generate_content(prompt)
    answer = response.text.strip()

    print(f"[RAGQueryAgent] Answer received ({len(answer)} characters).\n")
    return answer


# ── CrewAI Agent & Task factory ────────────────────────────────────────────────

def build_rag_query_agent() -> Agent:
    """Create and return the CrewAI RAG Query Agent."""
    return Agent(
        role="Research Synthesis Expert",
        goal=(
            "Retrieve the most relevant research passages from the Endee "
            "vector database and synthesise a grounded, cited answer to the "
            "user's research question using Google Gemini."
        ),
        backstory=(
            "You are a senior AI researcher with expertise in retrieval-augmented "
            "generation. You combine state-of-the-art vector search with large "
            "language models to provide accurate, evidence-backed answers to "
            "complex scientific questions."
        ),
        verbose=True,
        allow_delegation=False,
    )


def build_rag_query_task(agent: Agent, query: str) -> Task:
    """
    Build the CrewAI Task for the RAG Query Agent.

    Parameters
    ----------
    agent : Agent – The RAG query agent instance.
    query : str   – The user's original research question.
    """
    answer = rag_answer(query)  # uses default candidate_k=20, top_k=5

    return Task(
        description=textwrap.dedent(f"""
            Papers have been ingested into Endee. Now answer the user's question:
            "{query}"

            Steps already completed:
              1. Embedded the query using google/embeddinggemma-300m.
              2. Queried Endee for the top-5 most similar paper chunks.
              3. Sent the retrieved context to Google Gemini for synthesis.

            Deliver the final answer below to the user.
        """),
        agent=agent,
        expected_output="A detailed, cited research answer synthesised from retrieved ArXiv paper passages.",
        # Store the answer for crew.py to surface to the user
        _answer=answer,  # type: ignore[reportAttributeAccessIssue]
    )
