import os
import sys
sys.path.append(os.path.abspath('.'))

import json
import math
import asyncio
import logging
import random
from datetime import datetime
from pathlib import Path

import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_core.documents import Document
import pickle

# RAGAS imports (v0.4 collection metrics — NOT the deprecated evaluate())
from ragas.metrics.collections import ContextRecall, ContextPrecision
from ragas.llms import llm_factory
from ragas.cache import DiskCacheBackend
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CORPUS_PATH   = Path("evals/datasets/corpus.json")
RESULTS_DIR   = Path("evals/v1/results")
RAW_CACHE_DIR = Path("evals/v1/.raw_cache")  # cache inference results

CHROMA_PATH    = "chroma_db"
DOC_STORE_PATH = "doc_store_pdr"

JUDGE_MODEL   = "llama-3.3-70b-versatile"
EVAL_K        = 3            # retriever top-k
EVAL_TEMP     = 0.1          # chain LLM temperature

# Rate-limit delays (seconds)
CHAIN_DELAY   = 60      # between chain inference calls
METRIC_DELAY_MIN = 30.0    # between RAGAS judge calls
METRIC_DELAY_MAX = 60.0
SAMPLE_DELAY_MIN = 30.0      # between samples during scoring
SAMPLE_DELAY_MAX = 60.0


# ── Judge LLM ─────────────────────────────────────────────────────────────────
# Use AsyncOpenAI pointed at Groq's OpenAI-compatible endpoint.
# RAGAS's _patch_client_for_provider("groq") has a bug (calls
# client.messages.create, Anthropic-style).  Using provider="openai"
# with Groq's base_url avoids the issue entirely.
async_openai_client = openai.AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

cacher = DiskCacheBackend(cache_dir="./cache")
judge_llm = llm_factory(
    JUDGE_MODEL,
    client=async_openai_client,
    provider="openai",
    temperature=0,
    cache=cacher,
)

# ── Metrics (collection API — use ascore() directly) ──────────────────────────
METRICS = {
    "context_recall":    ContextRecall(llm=judge_llm),
    "context_precision": ContextPrecision(llm=judge_llm),
}
METRIC_NAMES = list(METRICS.keys())


def _is_rate_limit_error(exc: Exception) -> bool:
    """Retry only on rate-limit style failures (HTTP 429)."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    message = str(exc).lower()
    return "429" in message or "rate limit" in message


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(_is_rate_limit_error),
    reraise=True,
)
async def _score_metric_with_retry(metric_name: str, metric_obj, *, user_input: str, predicted: str, retrieved: list[str], reference: str):
    """
    Score one metric with backoff retry to handle transient 429 errors.
    """
    if metric_name == "context_recall":
        return await metric_obj.ascore(
            user_input=user_input,
            retrieved_contexts=retrieved,
            reference=reference,
        )
    if metric_name == "context_precision":
        return await metric_obj.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved,
        )
    raise ValueError(f"Unknown metric: {metric_name}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_embedding_model():
    """Load HuggingFace embedding model (no Streamlit dependency)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading embedding model BAAI/bge-m3 on %s...", device)
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_eval_retriever(k: int, embedding_model):
    """
    Build the same hybrid retriever used in qa_chain.py:
    - Chroma vector store over child chunks
    - BM25 retriever over all child docs
    - EnsembleRetriever combining both
    Also returns the parent-doc store for optional parent-level inspection.
    """
    vector_store = Chroma(
        collection_name="split_parents",
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )

    fs = LocalFileStore(DOC_STORE_PATH)
    doc_store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )

    child_vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    child_data = vector_store.get()
    all_child_docs = [
        Document(page_content=txt, metadata=md)
        for txt, md in zip(child_data["documents"], child_data["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(all_child_docs)
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[child_vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    return ensemble_retriever, doc_store


async def _run_retriever_only_inference(
    corpus: list[dict],
    retriever,
    doc_store,
) -> list[dict]:
    """
    Run hybrid retriever on each corpus item without calling the generator model.
    Caches results to disk so subsequent runs skip retrieval entirely when the
    corpus hasn't changed.
    """
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_CACHE_DIR / "raw_results.json"

    # Simple cache: reuse if corpus hasn't changed
    if cache_file.exists():
        cached = json.loads(cache_file.read_text("utf-8"))
        cached_ids = {r["_id"] for r in cached}
        corpus_ids = {item["id"] for item in corpus}
        if cached_ids == corpus_ids:
            log.info("♻️  Using cached retriever results (%d samples)", len(cached))
            return cached

    log.info("Running retriever-only inference for %d samples...", len(corpus))
    results = []

    for i, item in enumerate(corpus):
        query = item["user_input"]
        log.info("[%d/%d] Retrieving for: %s", i + 1, len(corpus), query[:80])

        try:
            # Retrieve child docs with the hybrid retriever
            child_docs = retriever.invoke(query)

            # Map to parent documents via doc_id metadata, mirroring qa_chain.py
            parent_ids = list({doc.metadata.get("doc_id") for doc in child_docs})
            parent_ids = [pid for pid in parent_ids if pid is not None]

            if parent_ids:
                parent_docs = doc_store.mget(parent_ids)
                valid_parents = [p for p in parent_docs if p is not None][:EVAL_K]
                retrieved_contexts = [
                    getattr(doc, "page_content", str(doc)) for doc in valid_parents
                ]
            else:
                retrieved_contexts = []
        except Exception as e:
            log.error("Retriever FAILED for sample %d: %s", item["id"], e)
            retrieved_contexts = []

        results.append({
            "_id":                item["id"],
            "user_input":         query,
            "predicted_response": "",                      # no generator in retriever-only mode
            "reference":          item["response"],        # ground truth answer
            "retrieved_contexts": retrieved_contexts,      # from hybrid retriever
            "reference_contexts": item.get("retrieved_contexts", ""),  # ground truth context
        })

        # Delay between retrieval calls (keep conservative to avoid DB thrash)
        if i < len(corpus) - 1:
            log.debug("Sleeping %ds before next retrieval...", CHAIN_DELAY)
            await asyncio.sleep(CHAIN_DELAY)

    # Save to disk for reuse
    cache_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Retriever-only inference cached → %s", cache_file)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    log.info("Starting RAGAS retriever-only evaluation pipeline")

    # ── Step 1: Load corpus ───────────────────────────────────────────────
    if not CORPUS_PATH.exists():
        log.error("Corpus not found: %s", CORPUS_PATH)
        return
    corpus = json.loads(CORPUS_PATH.read_text("utf-8"))
    log.info("Loaded %d samples from %s", len(corpus), CORPUS_PATH)

    # ── Step 2: Retriever-only inference ─────────────────────────────────
    embedding_model = _load_embedding_model()
    retriever, doc_store = _build_eval_retriever(EVAL_K, embedding_model)
    raw_results = await _run_retriever_only_inference(corpus, retriever, doc_store)

    if not raw_results:
        log.error("No samples collected — aborting evaluation.")
        return

    # ── Step 3: RAGAS scoring (direct ascore calls) ───────────────────────
    # Collection metrics expose ascore() directly — we loop samples × metrics.
    log.info(
        "Running RAGAS scoring — judge: %s, metrics: %s",
        JUDGE_MODEL, ", ".join(METRIC_NAMES),
    )

    per_sample = []
    all_scores: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}

    for idx, raw in enumerate(raw_results, 1):
        log.info(
            "[%d/%d] Scoring sample %d...", idx, len(raw_results), raw["_id"]
        )

        # Use predicted response + retrieved contexts from the chain
        predicted = raw["predicted_response"]
        retrieved = raw["retrieved_contexts"]
        reference = raw["reference"]          # ground truth answer
        user_input = raw["user_input"]

        # Ensure retrieved_contexts is a list of strings
        if isinstance(retrieved, str):
            retrieved = [retrieved] if retrieved else []

        sample_scores: dict[str, float | None] = {}

        for metric_name, metric_obj in METRICS.items():
            try:
                result = await _score_metric_with_retry(
                    metric_name,
                    metric_obj,
                    user_input=user_input,
                    predicted=predicted,
                    retrieved=retrieved,
                    reference=reference,
                )

                val = float(result.value)
                sample_scores[metric_name] = (
                    round(val, 4) if not math.isnan(val) else None
                )
                log.info("  ✓ %s = %s", metric_name, sample_scores[metric_name])

            except Exception as e:
                log.error(
                    "  ✗ %s FAILED for sample %d: %s",
                    metric_name, raw["_id"], e,
                )
                sample_scores[metric_name] = None

            # Delay between judge calls to respect Groq rate limits
            await asyncio.sleep(
                random.uniform(METRIC_DELAY_MIN, METRIC_DELAY_MAX)
            )

        # Accumulate for aggregates
        for m in METRIC_NAMES:
            v = sample_scores.get(m)
            if v is not None:
                all_scores[m].append(v)

        per_sample.append({
            "id":                 raw["_id"],
            "query":              user_input,
            "predicted_response": predicted,
            "reference":          reference,
            "retrieved_contexts": retrieved,
            "scores":             sample_scores,
        })

        # Delay between samples
        if idx < len(raw_results):
            delay = random.uniform(SAMPLE_DELAY_MIN, SAMPLE_DELAY_MAX)
            log.debug("Sleeping %.1fs before next sample...", delay)
            await asyncio.sleep(delay)

    # ── Step 4: Save results ──────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"

    aggregate = {
        m: round(sum(vals) / len(vals), 4) if vals else None
        for m, vals in all_scores.items()
    }

    output = {
        "metadata": {
            "timestamp":   timestamp,
            "judge_model": JUDGE_MODEL,
            "rag_k":       EVAL_K,
            "num_samples": len(raw_results),
        },
        "aggregate_scores": aggregate,
        "per_sample":       per_sample,
    }

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Results saved → %s", output_path)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n─── RAGAS Eval Summary ───────────────────────────────")
    for metric, score in aggregate.items():
        if score is not None:
            bar = "█" * int(score * 20)
            print(f"  {metric:<25} {score:.4f}  {bar}")
        else:
            print(f"  {metric:<25}   N/A")
    print(f"\n  Full results → {output_path}")
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    asyncio.run(main())