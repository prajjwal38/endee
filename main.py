"""
main.py
───────
Entry point for the Agentic RAG Pipeline.

Usage:
    python main.py
    python main.py --query "What are the latest advances in diffusion models?"
    python main.py --query "Transformer architectures for NLP" --papers 8
"""

from __future__ import annotations

import argparse
import os
import sys

# Silence TF/gRPC/ALTS startup noise before any library imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")          # Suppress TF C++ logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")         # Disable oneDNN verbose
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")            # Suppress gRPC info/warning
os.environ.setdefault("GRPC_TRACE", "")                     # No tracing

from dotenv import load_dotenv

# Load .env FIRST before any other project imports
load_dotenv()

from crew import run_pipeline  # noqa: E402 – must be after load_dotenv


BANNER = r"""
 ╔══════════════════════════════════════════════════════════════════╗
 ║          Agentic RAG Pipeline  –  Powered by Endee              ║
 ║  ArXiv   →   Endee (Vector DB)   →   Gemini   →   Answer       ║
 ╚══════════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG Pipeline: ArXiv → Endee → Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Research question (if omitted, enter interactively).",
    )
    parser.add_argument(
        "--papers", "-p",
        type=int,
        default=5,
        help="Maximum number of ArXiv papers to fetch (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    print(BANNER)

    args = parse_args()

    # ── Get query ──────────────────────────────────────────────────────────────
    if args.query:
        user_query = args.query.strip()
    else:
        print("Enter your research question (or 'quit' to exit):")
        user_query = input("  ▶ ").strip()

    if not user_query or user_query.lower() in {"quit", "exit", "q"}:
        print("Exiting. Goodbye!")
        sys.exit(0)

    # ── Run the pipeline ───────────────────────────────────────────────────────
    try:
        answer = run_pipeline(user_query, max_papers=args.papers)
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Pipeline error: {exc}")
        print("\nTroubleshooting tips:")
        print("  • Is Endee running?  →  docker compose up -d")
        print("  • Is GEMINI_API_KEY set in .env?")
        print("  • Did you run: pip install -r requirements.txt")
        sys.exit(1)

    # ── Print answer ───────────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  FINAL ANSWER  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝\n")
    print(answer)
    print()


if __name__ == "__main__":
    main()
