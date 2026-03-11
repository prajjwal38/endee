"""
agents/web_search_agent.py
──────────────────────────
Web Search Agent – fetches ArXiv papers relevant to a user query.

Responsibilities:
  • Accept a natural language research query.
  • Search the ArXiv API (via the `arxiv` Python package).
  • Return a structured list of papers: title, authors, abstract,
    PDF link, and ArXiv entry URL.

This module exposes both a plain Python function (`fetch_arxiv_papers`)
for use inside CrewAI tasks, and a CrewAI `Agent` + `Task` object pair
for use by the Crew coordinator in crew.py.
"""

from __future__ import annotations

import textwrap
import json
from typing import Any, Dict, List

import arxiv
from crewai import Agent, Task


# ── Core logic ─────────────────────────────────────────────────────────────────

def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search ArXiv for papers matching `query` and return structured metadata.

    Parameters
    ----------
    query       : str – Natural language or keyword research query.
    max_results : int – Maximum number of papers to retrieve (default 5).

    Returns
    -------
    List[Dict] with keys:
        title       (str)
        authors     (List[str])
        abstract    (str)
        pdf_url     (str)  – Direct link to the PDF
        entry_url   (str)  – ArXiv abstract page URL
        arxiv_id    (str)  – e.g. "2301.07041"
    """
    # Broadened search for science and physics
    enhanced_query = f"(physics OR science) AND ({query})"
    print(f"\n[WebSearchAgent] Searching ArXiv for: '{enhanced_query}' (max {max_results} papers) …")

    search = arxiv.Search(
        query=enhanced_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: List[Dict[str, Any]] = []
    client = arxiv.Client()

    for result in client.results(search):
        paper = {
            "title": result.title,
            "authors": [str(a) for a in result.authors],
            "abstract": result.summary.strip(),
            "pdf_url": result.pdf_url,
            "entry_url": result.entry_id,
            "arxiv_id": result.entry_id.split("/abs/")[-1],
        }
        papers.append(paper)
        print(f"  ✓ Found: {result.title[:80]} …")

    print(f"[WebSearchAgent] Retrieved {len(papers)} paper(s).\n")
    return papers


# ── CrewAI Agent & Task factory ────────────────────────────────────────────────

def build_web_search_agent() -> Agent:
    """Create and return the CrewAI Web Search Agent."""
    return Agent(
        role="ArXiv Research Retriever",
        goal=(
            "Find the most relevant academic papers from ArXiv that address "
            "the user's research question."
        ),
        backstory=(
            "You are an expert academic librarian with deep knowledge of "
            "scientific literature. You excel at identifying the most pertinent "
            "papers for any given research topic using the ArXiv database."
        ),
        verbose=True,
        allow_delegation=False,
    )


def build_web_search_task(agent: Agent, query: str) -> Task:
    """
    Build the CrewAI Task for the Web Search Agent.

    Parameters
    ----------
    agent : Agent – The web search agent instance.
    query : str   – The user's research question.
    """
    papers = fetch_arxiv_papers(query)
    papers_json = json.dumps(papers, indent=2)

    return Task(
        description=textwrap.dedent(f"""
            A user has asked the following research question:
            "{query}"

            You have already retrieved the following ArXiv papers:
            {papers_json}

            Your task: Summarise the retrieved papers and pass the full
            structured list to the next agent for ingestion.
            Output the raw JSON list of papers as your final answer.
        """),
        agent=agent,
        expected_output="A JSON array of ArXiv paper objects with title, abstract, pdf_url, entry_url, arxiv_id fields.",
        # Attach the fetched papers as context for downstream tasks
        context_json=papers_json,
        # Store papers on the task so crew.py can access them directly
        _papers=papers,  # type: ignore[reportAttributeAccessIssue]
    )
