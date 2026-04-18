# part1_rag_eval/src/data_loader.py
import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def safe_loader(
    source_dirs: List[Path],
    max_retsinformation_parts: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads chunks + embeddings from all source directories.
    Applies a part-file cap to retsinformation to prevent OOM.

    Returns:
        df_chunks     — all chunk rows, enriched with breadcrumbs from documents
        df_embeddings — embedding rows, enriched with breadcrumbs from chunks
    """
    all_docs, all_chunks, all_embs = [], [], []

    for base in source_dirs:
        base = Path(base)
        source_name = str(base).lower()

        # --- documents (for breadcrumb metadata) ---
        doc_path = base / "documents.parquet"
        if doc_path.exists():
            df_doc = pd.read_parquet(doc_path)
            keep = [c for c in ["document_id", "breadcrumbs", "l1_category", "title"] if c in df_doc.columns]
            all_docs.append(df_doc[keep])
        else:
            print(f"  [WARN] documents.parquet not found in {base}")

        # --- chunk parts ---
        limit = max_retsinformation_parts if "retsinformation" in source_name else 9999
        chunk_files = sorted(glob.glob(str(base / "chunks_part-*.parquet")))[:limit]
        for f in chunk_files:
            all_chunks.append(pd.read_parquet(f))
            print(f"  Loaded chunks: {os.path.basename(f)}")

        # --- embedding parts (optional — may be pre-joined elsewhere) ---
        emb_files = sorted(glob.glob(str(base / "embeddings_part-*.parquet")))[:limit]
        for f in emb_files:
            all_embs.append(pd.read_parquet(f))
            print(f"  Loaded embeddings: {os.path.basename(f)}")

    if not all_chunks:
        raise ValueError("No chunk files found. Check DATA_DIR and source paths in params.yaml.")

    df_docs   = pd.concat(all_docs, ignore_index=True).drop_duplicates("document_id") if all_docs else pd.DataFrame()
    df_chunks = pd.concat(all_chunks, ignore_index=True)

    # Enrich chunks with doc metadata
    if not df_docs.empty:
        df_chunks = df_chunks.merge(df_docs, on="document_id", how="left")

    if all_embs:
        df_embeddings = pd.concat(all_embs, ignore_index=True)
        # Enrich embeddings with breadcrumbs from chunks
        bc_cols = [c for c in ["chunk_id", "breadcrumbs", "l1_category"] if c in df_chunks.columns]
        df_embeddings = df_embeddings.merge(df_chunks[bc_cols], on="chunk_id", how="left")
    else:
        # Fallback: embeddings may be embedded inside the chunks df itself
        if "embedding" in df_chunks.columns:
            print("[INFO] No separate embedding files found — using 'embedding' column from chunks.")
            df_embeddings = df_chunks[["chunk_id", "embedding"] + [
                c for c in ["breadcrumbs", "l1_category"] if c in df_chunks.columns
            ]].copy()
        else:
            raise ValueError(
                "No embedding files found and no 'embedding' column in chunks. "
                "Place embedding parquets alongside chunk parquets, or add an 'embedding' column."
            )

    print(f"\n[DATA] {len(df_chunks):,} chunks | {len(df_embeddings):,} embedding rows loaded.")
    return df_chunks, df_embeddings


def preprocess_embeddings(
    df: pd.DataFrame,
    vector_col: str = "embedding",
) -> Tuple[List[str], np.ndarray]:
    """
    Stacks the embedding column into a float32 numpy array for FAISS.

    Returns:
        chunk_ids     — list of chunk_id strings (same order as vectors)
        vectors       — (N, D) float32 array
    """
    vectors   = np.stack(df[vector_col].values).astype("float32")
    chunk_ids = df["chunk_id"].tolist()
    return chunk_ids, vectors
