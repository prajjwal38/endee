# Agentic RAG Pipeline

> **ArXiv Research Papers → Endee Vector DB → Google Gemini → Grounded Answers**

A fully agentic Retrieval-Augmented Generation system that answers research questions by fetching, ingesting, and querying ArXiv papers — all coordinated by [CrewAI](https://github.com/joaomdmoura/crewAI).

---

## Problem Statement

Large language models hallucinate when asked about recent or niche research topics because their training data has a cutoff date. This pipeline solves that by:

1. **Fetching live papers** from ArXiv at query time.
2. **Storing them as vector embeddings** in the Endee vector database.
3. **Grounding every answer** by retrieving the most relevant passages before calling Gemini.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER RESEARCH QUERY                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│             Agent 1: Web Search Agent (CrewAI)              │
│                                                             │
│  ArXiv API ──► Fetch top-N papers                           │
│  Output: [{ title, abstract, source_tarball_url, etc. }]    │
└────────────────────────────┬────────────────────────────────┘
                             │  papers list
                             ▼
┌─────────────────────────────────────────────────────────────┐
│            Agent 2: Ingestion Agent (CrewAI)                │
│                                                             │
│  LaTeX Download (tarball) → fallback to PDF/abstract        │
│  Ollama Semantic Chunker  → gpt-oss:120b semantic boundaries│
│  EmbeddingGemma-300m      → 768-dim dense vectors           │
│  Endee SDK                → upsert with enriched metadata   │
└────────────────────────────┬────────────────────────────────┘
                             │  N chunks stored in Endee
                             ▼
┌─────────────────────────────────────────────────────────────┐
│           Agent 3: RAG Query Agent (CrewAI)                 │
│                                                             │
│  Gemini 2.5 Flash     → pre-retrieval query expansion       │
│  EmbeddingGemma-300m  → embed 5 variants & average vector   │
│  Endee Query          → top-20 ANN candidates (cosine)      │
│  Qwen3-Reranker-0.6B  → re-score & re-rank to top-5 (opt)  │
│  Context Assembly     → ranked chunks + metadata            │
│  Gemini 2.5 Flash     → grounded, cited answer              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │  CITED ANSWER    │
                   │  [Source 1] ...  │
                   │  [Source 2] ...  │
                   │  References: ... │
                   └──────────────────┘
```

---

## Stack

| Component        | Technology                                        |
|-----------------|---------------------------------------------------|
| Agent Framework  | [CrewAI](https://github.com/joaomdmoura/crewAI)  |
| Vector Database  | [Endee](https://endee.io) (Docker, localhost:8080) |
| Embeddings       | `google/embeddinggemma-300m` via sentence-transformers (768-dim) |
| Semantic Chunker | Ollama locally (`gpt-oss:120b` turbo mode)       |
| Document Parser  | LaTeX source extraction (tarballs) + PyMuPDF fallback |
| Re-ranker        | `Qwen/Qwen3-Reranker-0.6B` via transformers (optional) |
| LLM              | Google Gemini 2.5 Flash                          |
| Paper Source     | ArXiv API (`arxiv` Python package)               |

---

## How Endee Is Used

Endee is an open-source, high-performance vector database that runs locally via Docker.

In this project:
- **Index**: `arxiv_papers` — created at startup (cosine similarity, 768 dims).
- **Upsert**: Each paper is semantically chunked by Ollama. Every chunk is embedded with EmbeddingGemma and upserted with enriched metadata (`title`, `authors`, `sub_heading`, `primary_category`, `source_type`, `text`, etc.).
- **Query**: The user's question is expanded into multiple variants, averaged into a single query vector, and sent to Endee's `/query` endpoint. The top-20 candidates are pulled and then either reranked or fed directly to Gemini.

```python
# Connecting to Endee
from endee import Client
client = Client(url="http://localhost:8080")

# Creating the index
client.create_index(name="arxiv_papers", dimension=768, space_type="cosine")

# Upserting a vector
client.upsert(index_name="arxiv_papers", vectors=[{
    "id": "2301.07041_chunk_0",
    "vector": [...],   # 768-dim float32 list
    "meta": { 
        "title": "...", 
        "chunk_index": 0, 
        "sub_heading": "Introduction",
        "primary_category": "gr-qc",
        "source_type": "latex",
        "text": "..." 
    }
}])

# Querying
results = client.query(index_name="arxiv_papers", vector=[...], top_k=5)
```

---

## Project Structure

```
endee/
├── agents/
│   ├── __init__.py            # Package re-exports
│   ├── web_search_agent.py    # Agent 1: ArXiv fetcher
│   ├── ingestion_agent.py     # Agent 2: PDF → Endee
│   └── rag_query_agent.py     # Agent 3: Endee → Gemini
├── crew.py                    # CrewAI sequential coordinator
├── embeddings.py              # EmbeddingGemma-300m helper
├── endee_client.py            # Endee SDK wrapper
├── main.py                    # CLI entry point
├── requirements.txt
├── docker-compose.yml         # Endee Docker config (pre-existing)
└── .env                       # Secrets (not committed)
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- Docker Desktop running
- A [Google AI Studio](https://aistudio.google.com/) API key

### 2. Start Endee via Docker

```bash
docker compose up -d
```

Verify it's running:
```bash
curl http://localhost:8080/health
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `google/embeddinggemma-300m` requires `sentence-transformers>=2.2.2` and will be downloaded (~600 MB) from HuggingFace on the first run. You may need to accept the model license on HuggingFace.

### 4. Configure Environment Variables

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
ENDEE_URL=http://localhost:8080
ENDEE_AUTH_TOKEN=          # leave blank if you didn't set NDD_AUTH_TOKEN

ENABLE_RERANKER=false      # optionally enable Qwen reranking
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b
```

### 5. Run the Pipeline

**Interactive mode:**
```bash
python main.py
```

**With CLI arguments:**
```bash
python main.py --query "What are the latest advances in diffusion models for image generation?" --papers 5
```

---

## Example Query & Output

**Query:**
```
What are the latest advances in diffusion models for image generation?
```

**Pipeline execution:**
```
═══════════════════════════════════════════════════════════════════════
  Agentic RAG Pipeline  –  Starting
═══════════════════════════════════════════════════════════════════════
  Query : What are the latest advances in diffusion models for image generation?
  Papers: up to 5 from ArXiv

▶ Phase 1/3 — Web Search Agent
[WebSearchAgent] Searching ArXiv: 'diffusion models image generation' …
  ✓ Found: Denoising Diffusion Probabilistic Models …
  ✓ Found: Stable Diffusion: High-Resolution Image Synthesis …
  ✓ Found: DALL-E 3: Improving Image Generation with …
  ✓ Found: DiT: Scalable Diffusion Models with Transformers …
  ✓ Found: Consistency Models …
[WebSearchAgent] Retrieved 5 paper(s).

▶ Phase 2/3 — Ingestion Agent
[IngestionAgent] Processing: Denoising Diffusion Probabilistic Models …
  [Ingestion] PDF extracted: 82,341 characters.
  [Ingestion] 47 chunk(s) generated.
[EndeeClient]  Upserting 47 vectors …
... (repeated for each paper)
  ✓ 198 chunk(s) stored in Endee.

▶ Phase 3/3 — RAG Query Agent
[Embeddings]  Loading model 'google/embeddinggemma-300m' …
[EndeeClient] Querying 'arxiv_papers' for top-5 matches …
[RAGQueryAgent] Sending context + query to Gemini (gemini-1.5-flash) …
```

**Final Answer (excerpt):**
```
Recent advances in diffusion models for image generation have focused on
three key areas:

**1. Architecture Improvements**
Diffusion Transformers (DiT) [Source 4] replace the traditional U-Net
backbone with a Transformer architecture, showing superior scaling properties
and state-of-the-art FID scores on ImageNet…

**2. Efficiency and Consistency**
Consistency Models [Source 5] enable single-step generation by training
models to map any point on a diffusion trajectory directly to its origin,
achieving 10-80x faster inference compared to DDPM…

**References**
- [Source 1] Denoising Diffusion Probabilistic Models — https://arxiv.org/abs/2006.11239
- [Source 4] DiT: Scalable Diffusion Models with Transformers — https://arxiv.org/abs/2212.09748
- [Source 5] Consistency Models — https://arxiv.org/abs/2303.01469
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Connection refused localhost:8080` | Run `docker compose up -d` |
| `GEMINI_API_KEY not set` | Add key to `.env` file |
| `HuggingFace 401 error` | Accept the EmbeddingGemma license at huggingface.co/google/embeddinggemma-300m |
| `float16` error with EmbeddingGemma | Already handled — we force `float32` in `embeddings.py` |
| PDF download fails | Ingestion agent automatically falls back to the paper abstract |
| `transformers` version error with reranker | Run `pip install "transformers>=4.51.0"` — Qwen3-Reranker requires this minimum version |

---

## License

This project is provided for educational and demonstration purposes.
The EmbeddingGemma model is subject to [Google's model license](https://huggingface.co/google/embeddinggemma-300m).
