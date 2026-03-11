"""
agents/__init__.py
──────────────────
Makes `agents` a Python package and re-exports the key builder functions
so crew.py can import them cleanly.
"""

from agents.web_search_agent import build_web_search_agent, build_web_search_task, fetch_arxiv_papers
from agents.ingestion_agent import build_ingestion_agent, build_ingestion_task, ingest_papers
from agents.rag_query_agent import build_rag_query_agent, build_rag_query_task, rag_answer

__all__ = [
    "build_web_search_agent",
    "build_web_search_task",
    "fetch_arxiv_papers",
    "build_ingestion_agent",
    "build_ingestion_task",
    "ingest_papers",
    "build_rag_query_agent",
    "build_rag_query_task",
    "rag_answer",
]
