"""
reranker.py
───────────
Qwen3-Reranker-0.6B re-ranking module.

Takes a user query and a list of candidate chunks retrieved from Endee
(via ANN search) and re-scores each (query, document) pair using the
Qwen3-Reranker-0.6B cross-encoder.  Returns the candidates sorted by
relevance score, trimmed to `top_k`.

Model details
─────────────
  • Model    : Qwen/Qwen3-Reranker-0.6B
  • Type     : Generative cross-encoder (AutoModelForCausalLM)
  • Scoring  : log-softmax on the final-token logits for "yes" vs "no"
  • Requires : transformers >= 4.51.0
  • Precision: float32 (CPU-safe; GPU used automatically if available)
  • Size     : ~1.2 GB download on first run
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Model configuration ────────────────────────────────────────────────────────
RERANKER_MODEL: str = "Qwen/Qwen3-Reranker-0.6B"

# Task instruction sent to the reranker (customises relevance judgement)
RERANK_TASK: str = (
    "Given a scientific research question, judge whether the document "
    "is relevant and helpful in answering the question."
)

# Maximum token length for (query + document) pairs
MAX_LENGTH: int = 8192

# ── Singleton state ────────────────────────────────────────────────────────────
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_token_true_id: Optional[int] = None
_token_false_id: Optional[int] = None
_prefix_tokens: Optional[List[int]] = None
_suffix_tokens: Optional[List[int]] = None


# ── Lazy loader ────────────────────────────────────────────────────────────────

def _load_reranker() -> None:
    """
    Load the Qwen3-Reranker-0.6B tokenizer and model (once per process).
    Sets up the chat-format prefix/suffix tokens used for scoring.
    """
    global _tokenizer, _model, _token_true_id, _token_false_id
    global _prefix_tokens, _suffix_tokens

    if _model is not None:
        return  # Already loaded

    print(f"[Reranker] Loading {RERANKER_MODEL} — first run may take a moment …")

    _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, padding_side="left")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = AutoModelForCausalLM.from_pretrained(
        RERANKER_MODEL,
        dtype=torch.float32,   # float32 for CPU safety
    ).to(device).eval()

    # Token IDs for the yes/no judgement tokens
    # Note: convert_tokens_to_ids may return None for subword-split tokens.
    # Using encode() is the reliable way to get the token ID.
    _token_true_id = _tokenizer.encode("yes", add_special_tokens=False)[0]
    _token_false_id = _tokenizer.encode("no", add_special_tokens=False)[0]

    # Build the chat prefix/suffix that wraps every (query, doc) pair
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = (
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    _prefix_tokens = _tokenizer.encode(prefix, add_special_tokens=False)
    _suffix_tokens = _tokenizer.encode(suffix, add_special_tokens=False)

    print(f"[Reranker] Model loaded on {device}.")


# ── Prompt formatting ──────────────────────────────────────────────────────────

def _format_pair(query: str, document: str, instruction: str = RERANK_TASK) -> str:
    """Format a (query, document) pair using the Qwen3-Reranker prompt template."""
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


# ── Tokenisation and scoring ───────────────────────────────────────────────────

def _tokenize_pairs(pairs: List[str]) -> Dict[str, Any]:
    """
    Tokenise a batch of complete prompt strings (prefix + body + suffix baked in).
    Using a single __call__ is faster and avoids the Qwen fast-tokenizer warning
    that arises from encoding + padding as two separate steps.
    """
    # Decode prefix / suffix tokens back to strings so we can build the full prompt
    prefix_str = _tokenizer.decode(_prefix_tokens, skip_special_tokens=False)
    suffix_str = _tokenizer.decode(_suffix_tokens, skip_special_tokens=False)

    full_prompts = [prefix_str + p + suffix_str for p in pairs]

    encoded = _tokenizer(
        full_prompts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    device = next(_model.parameters()).device
    return {k: v.to(device) for k, v in encoded.items()}


@torch.no_grad()
def _score_pairs(pairs: List[str]) -> List[float]:
    """
    Score a list of formatted (query, doc) pairs.
    Returns a float score in [0, 1] for each pair (higher = more relevant).
    """
    inputs = _tokenize_pairs(pairs)
    logits = _model(**inputs).logits[:, -1, :]          # last token logits

    true_vec = logits[:, _token_true_id]
    false_vec = logits[:, _token_false_id]
    stacked = torch.stack([false_vec, true_vec], dim=1)
    log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
    scores = log_probs[:, 1].exp().tolist()              # probability of "yes"
    return scores


# ── Public API ─────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    Re-rank a list of Endee result dicts using Qwen3-Reranker-0.6B.

    Parameters
    ----------
    query      : str              – The user's research question.
    candidates : List[Dict]       – Raw results from `endee_client.query_index()`.
                                    Each dict must have a ``metadata.text`` field.
    top_k      : int              – Number of top candidates to return (default 5).
    batch_size : int              – Scoring batch size (lower = less VRAM, default 8).

    Returns
    -------
    List[Dict]  – The top-k candidates sorted by Qwen relevance score (descending).
                  Each dict gains a ``rerank_score`` key alongside the original fields.
    """
    if not candidates:
        return candidates

    _load_reranker()

    print(f"[Reranker] Scoring {len(candidates)} candidate(s) …")

    # Collect all formatted pair strings
    pairs = []
    for cand in candidates:
        doc_text = cand.get("metadata", {}).get("text", "") or ""
        pairs.append(_format_pair(query, doc_text))

    # Score in batches
    all_scores: List[float] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        all_scores.extend(_score_pairs(batch))

    # Attach scores and sort
    for cand, score in zip(candidates, all_scores):
        cand["rerank_score"] = score

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    top = ranked[:top_k]

    print(
        f"[Reranker] Top-{top_k} after rerank | "
        + " | ".join(
            f"{c.get('id', '?')} → {c['rerank_score']:.4f}" for c in top
        )
    )

    return top
