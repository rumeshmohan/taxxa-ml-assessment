#!/usr/bin/env python3
"""
Part 1 — RAG Evaluation Pipeline
Run from the project root:  python part1_rag_eval/run.py

Env vars:
  DATA_DIR          path to parquet data (default: ./data)
  SKIP_GENERATION=1 skip LLM answer generation
  SKIP_RAGAS=1      skip RAGAS scoring
"""

import os
import sys
import time
import logging
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from utils.config       import PARAMS, resolve_rag_data_paths, get_active_generator_config
from utils.llm_services import LLMService
from part1_rag_eval.src.data_loader  import safe_loader, preprocess_embeddings
from part1_rag_eval.src.retrievers   import DenseRetriever, BM25Retriever, HybridRetriever
from part1_rag_eval.src.graph_utils  import MetadataGraphRetriever
from part1_rag_eval.src.metrics      import EvaluationTracker, evaluate_ragas

_LOG_PATH = _ROOT / PARAMS.get("rag", {}).get("output", {}).get("log_file", "part1_rag_eval/outputs/part1_run.log")
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def banner(msg: str):
    log.info("=" * 60)
    log.info(f"  {msg}")
    log.info("=" * 60)


def step_load_data():
    banner("STEP 1 — Loading data")
    paths = resolve_rag_data_paths()
    log.info(f"DATA_DIR : {os.getenv('DATA_DIR', '<default: ./data>')}")
    log.info(f"Sources  : {[str(p) for p in paths['source_dirs']]}")
    log.info(f"Queries  : {paths['queries_path']}")

    df_chunks, df_embeddings = safe_loader(
        source_dirs=paths["source_dirs"],
        max_retsinformation_parts=paths["max_rets"],
    )

    if not paths["queries_path"].exists():
        raise FileNotFoundError(
            f"Queries file not found: {paths['queries_path']}\n"
            "Set DATA_DIR or check params.yaml → rag.data.queries_file"
        )
    df_queries = pd.read_parquet(paths["queries_path"])
    log.info(f"Queries  : {len(df_queries)} rows | cols: {df_queries.columns.tolist()}")

    chunk_ids, corpus_vectors = preprocess_embeddings(df_embeddings, "embedding")
    id_to_text = dict(zip(df_chunks["chunk_id"], df_chunks["chunk_text"]))

    # Build chunk_id → document URL for ground-truth matching.
    # The assessment scores by document_url, not chunk_id, so retrieved chunk_ids
    # must be resolved to their parent document's URL before recall is computed.
    if "url" in df_chunks.columns:
        chunk_to_url = dict(zip(df_chunks["chunk_id"], df_chunks["url"]))
        log.info("chunk_to_url built from 'url' column ✓")
    elif "document_id" in df_chunks.columns:
        log.warning("No 'url' in chunks — falling back to document_id for recall matching.")
        chunk_to_url = dict(zip(df_chunks["chunk_id"], df_chunks["document_id"]))
    else:
        chunk_to_url = {}
        log.warning("Cannot build chunk_to_url — recall metrics will be 0.")

    log.info(f"Corpus   : {len(df_embeddings):,} vectors  dim={corpus_vectors.shape[1]}")
    return df_chunks, df_embeddings, df_queries, chunk_ids, corpus_vectors, id_to_text, chunk_to_url


def step_build_retrievers(df_chunks, df_embeddings, corpus_vectors, chunk_ids):
    banner("STEP 2 — Building retrievers")
    rag_cfg = PARAMS.get("rag", {})

    dense  = DenseRetriever(corpus_vectors, chunk_ids, rag_cfg.get("embedding_dim", 3072))
    log.info("Dense   (FAISS IndexFlatIP) ✓")

    bm25   = BM25Retriever(df_chunks["chunk_text"].tolist(), df_chunks["chunk_id"].tolist())
    log.info("BM25    (BM25Okapi) ✓")

    hybrid = HybridRetriever(dense, bm25, rrf_k=rag_cfg.get("rrf_k", 60))
    log.info(f"Hybrid  (RRF k={rag_cfg.get('rrf_k', 60)}) ✓")

    graph  = MetadataGraphRetriever(base_retriever=hybrid, df_corpus=df_embeddings)
    log.info("Graph   (breadcrumb MetadataGraph over Hybrid) ✓")

    return {"Dense": dense, "Hybrid": hybrid, "Graph": graph}


def step_evaluate(retrievers, df_queries, id_to_text, chunk_to_url, llm_client):
    banner("STEP 3 — Retrieval + Generation evaluation")
    top_k = PARAMS.get("rag", {}).get("top_k", 10)

    # Prefer URL-based ground truth (gold_document_url); fall back to any id column.
    url_cols = [c for c in df_queries.columns if "url" in c.lower()]
    id_cols  = [c for c in df_queries.columns if "id" in c.lower() and "query" not in c.lower()]
    gt_col   = url_cols[0] if url_cols else (id_cols[0] if id_cols else None)
    if gt_col is None:
        raise ValueError(f"Ground-truth column not found. Columns: {df_queries.columns.tolist()}")
    log.info(f"Ground-truth column: '{gt_col}'")

    use_url_matching = "url" in gt_col.lower() and bool(chunk_to_url)

    trackers   = {n: EvaluationTracker() for n in retrievers}
    ragas_data = {n: {"questions": [], "answers": [], "contexts": []} for n in retrievers}

    total = len(df_queries)
    for q_idx, (_, row) in enumerate(df_queries.iterrows(), start=1):
        q_vec  = row["embedding"]
        q_text = row.get("query_text", "")
        gt     = [row[gt_col]]
        log.info(f"  Query {q_idx}/{total}: {q_text[:80]!r}")

        for name, retriever in retrievers.items():
            t0  = time.perf_counter()
            res = (retriever.search(q_vec, top_k=top_k) if name == "Dense"
                   else retriever.search(q_vec, q_text, top_k=top_k))
            retrieved_ids  = [rid for rid, _ in res]
            retrieval_time = time.perf_counter() - t0

            # Resolve chunk_ids → document URLs to match the ground-truth ID space.
            if use_url_matching:
                resolved_ids = [chunk_to_url.get(cid, cid) for cid in retrieved_ids]
            else:
                resolved_ids = retrieved_ids

            trackers[name].record_query(resolved_ids, gt, retrieval_time)

            ctx_texts = [id_to_text.get(cid, "") for cid in retrieved_ids[:5]]
            answer    = ""
            if llm_client:
                prompt = (
                    f"Context:\n{chr(10).join(ctx_texts)}\n\n"
                    f"Question: {q_text}\nAnswer accurately based only on the context:"
                )
                answer = llm_client.generate_text(
                    prompt,
                    system_prompt="You are a helpful Danish accounting and tax assistant.",
                )
            ragas_data[name]["questions"].append(q_text)
            ragas_data[name]["answers"].append(answer)
            ragas_data[name]["contexts"].append(ctx_texts)

    return trackers, ragas_data


def step_compile_results(trackers, ragas_data):
    banner("STEP 4 — Computing metrics & saving results")
    cost_per_query = 0.0026 / 1000
    skip_ragas     = os.getenv("SKIP_RAGAS", "0") == "1"

    results = []
    for name, tracker in trackers.items():
        summary = tracker.get_summary(cost_per_query_usd=cost_per_query)
        summary["Strategy"] = name

        if not skip_ragas:
            log.info(f"[RAGAS] Running for {name} …")
            try:
                judge = PARAMS.get("active_generator", {}).get("model", "llama3.1:8b")
                r = evaluate_ragas(
                    questions=ragas_data[name]["questions"],
                    answers=ragas_data[name]["answers"],
                    contexts=ragas_data[name]["contexts"],
                    ollama_model=judge,
                )
                summary["Faithfulness"]     = round(r.get("faithfulness", 0.0), 4)
                summary["Answer_Relevancy"] = round(r.get("answer_relevancy", 0.0), 4)
            except Exception as exc:
                log.warning(f"[RAGAS] Failed for {name}: {exc}")
                summary["Faithfulness"] = summary["Answer_Relevancy"] = None
        else:
            log.info("[RAGAS] Skipped (SKIP_RAGAS=1)")
            summary["Faithfulness"] = summary["Answer_Relevancy"] = None

        results.append(summary)

    COLS = ["Strategy", "Recall@1", "Recall@5", "Recall@10", "nDCG@10",
            "Faithfulness", "Answer_Relevancy",
            "p50_latency_sec", "p95_latency_sec", "cost_per_1k_queries_usd"]
    df = pd.DataFrame(results)
    df = df[[c for c in COLS if c in df.columns]]

    out = _ROOT / PARAMS.get("rag", {}).get("output", {}).get("results_csv", "part1_rag_eval/outputs/part1_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    log.info(f"\n{df.to_string(index=False)}")
    log.info(f"\nResults saved → {out}")
    return df


def main():
    banner("Part 1 — RAG Evaluation Pipeline")

    df_chunks, df_embeddings, df_queries, chunk_ids, corpus_vectors, id_to_text, chunk_to_url = step_load_data()
    retrievers = step_build_retrievers(df_chunks, df_embeddings, corpus_vectors, chunk_ids)

    llm_client = None
    if os.getenv("SKIP_GENERATION", "0") != "1":
        try:
            cfg        = get_active_generator_config()
            llm_client = LLMService(cfg["base_url"], cfg["api_key"], cfg["model_name"])
            log.info(f"LLM: {cfg['provider'].upper()} / {cfg['model_name']}")
        except Exception as exc:
            log.warning(f"LLM init failed ({exc}). Generation skipped.")
    else:
        log.info("Generation skipped (SKIP_GENERATION=1).")

    trackers, ragas_data = step_evaluate(retrievers, df_queries, id_to_text, chunk_to_url, llm_client)
    step_compile_results(trackers, ragas_data)
    banner("Done ✓")


if __name__ == "__main__":
    main()