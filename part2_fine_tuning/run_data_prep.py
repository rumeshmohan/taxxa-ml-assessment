#!/usr/bin/env python3
"""
Part 2 — Step 1: Data Preparation
Step 1 of Part 2: Load Kila chunks → deduplicate → document-level train/val
split → generate synthetic Finnish Q&A pairs via LLM → save as JSONL.

Run from project root:
    python part2_fine_tuning/run_data_prep.py

Env vars:
    DATA_DIR          path to parquet data (default: ./data)
    RESUME_TRAIN=N    skip first N training chunks (resume after crash)
    RESUME_VAL=N      skip first N val chunks
    DRY_RUN=1         generate pairs for 3 chunks only (sanity check)
"""

import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from utils.config         import PARAMS, resolve_ft_data_paths, get_active_generator_config
from utils.llm_services   import LLMService
from part2_fine_tuning.src.dataset_generator import ModelAgnosticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FT  = PARAMS.get("finetuning", {})
DAT = FT.get("data", {})


def banner(msg: str):
    log.info("=" * 60)
    log.info(f"  {msg}")
    log.info("=" * 60)


def load_kila_chunks(chunk_dirs) -> pd.DataFrame:
    banner("STEP 1A — Loading Kila chunks")
    all_chunks, all_docs = [], []

    for base in chunk_dirs:
        base = Path(base)
        doc_path = base / "documents.parquet"
        if doc_path.exists():
            df_doc = pd.read_parquet(doc_path)
            all_docs.append(df_doc[["document_id", "l1_category"]])

        for f in sorted(glob.glob(str(base / "chunks_part-*.parquet"))):
            df = pd.read_parquet(f)
            all_chunks.append(df)
            log.info(f"  Loaded: {Path(f).name}  ({len(df):,} rows)")

    if not all_chunks:
        raise FileNotFoundError(
            "No Kila chunk files found. Check DATA_DIR and params.yaml → "
            "finetuning.data.chunks_sources"
        )

    df_raw = pd.concat(all_chunks, ignore_index=True)
    if all_docs:
        df_docs = pd.concat(all_docs, ignore_index=True).drop_duplicates("document_id")
        df_raw  = df_raw.merge(df_docs, on="document_id", how="left")

    log.info(f"Raw chunks: {len(df_raw):,} | columns: {df_raw.columns.tolist()}")
    return df_raw


def dedupe_and_filter(df_raw: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 1B — Deduplication & filtering")
    min_len = DAT.get("min_chunk_len", 120)

    before = len(df_raw)
    df = df_raw.drop_duplicates(subset=["chunk_hash"]).copy()
    log.info(f"After chunk_hash dedup : {len(df):,} (removed {before - len(df):,})")

    df = df[df["chunk_text"].str.len() >= min_len].copy()
    log.info(f"After length filter    : {len(df):,} (min {min_len} chars)")

    df = df.dropna(subset=["chunk_text", "document_id"]).reset_index(drop=True)
    log.info(f"After null drop        : {len(df):,}")
    log.info(f"Unique documents       : {df['document_id'].nunique():,}")
    return df


def split_train_val(df: pd.DataFrame, paths):
    banner("STEP 1C — Document-level train/val split")
    val_ratio   = DAT.get("val_ratio", 0.10)
    random_seed = DAT.get("random_seed", 42)

    all_doc_ids = df["document_id"].unique()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(all_doc_ids)

    n_val        = max(1, int(len(all_doc_ids) * val_ratio))
    val_doc_ids  = set(all_doc_ids[:n_val])
    train_doc_ids = set(all_doc_ids[n_val:])

    df_train = df[df["document_id"].isin(train_doc_ids)].reset_index(drop=True)
    df_val   = df[df["document_id"].isin(val_doc_ids)].reset_index(drop=True)

    log.info(f"Train docs: {len(train_doc_ids):,}  | train chunks: {len(df_train):,}")
    log.info(f"Val   docs: {len(val_doc_ids):,}  | val   chunks: {len(df_val):,}")

    # Persist val doc IDs for reproducible evaluation later
    val_ids_path = paths["val_doc_ids_path"]
    val_ids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_ids_path, "w", encoding="utf-8") as f:
        json.dump(list(val_doc_ids), f)
    log.info(f"Val doc IDs saved → {val_ids_path}")

    return df_train, df_val


def run_generation(df: pd.DataFrame, output_path: Path, generator, resume_idx: int = 0):
    batch_size  = DAT.get("batch_size", 15)
    batch_delay = DAT.get("batch_delay_sec", 20)
    rpm_delay   = DAT.get("rpm_delay_sec", 2.1)
    dry_run     = os.getenv("DRY_RUN", "0") == "1"

    df_slice = df.iloc[resume_idx:]
    total    = len(df_slice) if not dry_run else 3
    if dry_run:
        df_slice = df_slice.head(3)
        log.info("[DRY RUN] Processing 3 chunks only.")

    log.info(f"Starting from chunk {resume_idx} | remaining: {total:,}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs   = 0
    failed_chunks = 0

    for loop_idx, (_, row) in enumerate(tqdm(df_slice.iterrows(), total=total), start=1):
        try:
            pairs = generator.generate_qa_pairs(row["chunk_text"])
            if pairs:
                generator.save_to_jsonl(pairs, str(output_path))
                total_pairs += len(pairs)
            else:
                failed_chunks += 1
        except Exception as exc:
            failed_chunks += 1
            log.warning(f"Chunk {resume_idx + loop_idx} failed: {exc}")

        time.sleep(rpm_delay)

        if loop_idx % batch_size == 0:
            log.info(f"[BATCH] {loop_idx} chunks done. Sleeping {batch_delay}s …")
            time.sleep(batch_delay)

    log.info(f"Pairs generated: {total_pairs:,} | failed chunks: {failed_chunks:,}")
    log.info(f"Output → {output_path}")
    return total_pairs


def validate_jsonl(path: Path, label: str = ""):
    if not path.exists():
        log.warning(f"[VALIDATE] {path} not found.")
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    valid, errors = 0, 0
    for i, line in enumerate(lines):
        try:
            json.loads(line)
            valid += 1
        except json.JSONDecodeError:
            errors += 1
            log.warning(f"  Invalid JSON at line {i+1}: {line[:60]}")

    sample = "".join(lines[:5])
    broken_utf8 = any(c in sample for c in ["Ã¤", "Ã¶", "Ã¥"])
    log.info(f"[VALIDATE {label}] {valid:,} valid pairs | {errors} errors | "
             f"UTF-8 ok: {not broken_utf8}")


def main():
    banner("Part 2 — Step 1: Data Preparation")

    paths     = resolve_ft_data_paths()
    df_raw    = load_kila_chunks(paths["chunk_dirs"])
    df_clean  = dedupe_and_filter(df_raw)
    df_train, df_val = split_train_val(df_clean, paths)

    # Init LLM + generator
    llm_cfg   = get_active_generator_config()
    llm       = LLMService(llm_cfg["base_url"], llm_cfg["api_key"], llm_cfg["model_name"])
    generator = ModelAgnosticDataGenerator(llm_service=llm)
    log.info(f"LLM: {llm_cfg['provider'].upper()} / {llm_cfg['model_name']}")

    # Sanity check on one chunk
    sample = df_train["chunk_text"].iloc[0]
    test   = generator.generate_qa_pairs(sample)
    log.info(f"Sanity check: {len(test)} pair(s) from first chunk.")
    for p in test:
        log.info(f"  Q: {p['instruction']}")
        log.info(f"  A: {p['response'][:120]} …")

    # Generate training set (supports resume via RESUME_TRAIN env var)
    banner("STEP 1D-TRAIN — Generating training pairs")
    resume_train = int(os.getenv("RESUME_TRAIN", "0"))
    run_generation(df_train, paths["train_jsonl"], generator, resume_idx=resume_train)
    validate_jsonl(paths["train_jsonl"], "TRAIN")

    # Generate validation set
    banner("STEP 1D-VAL — Generating validation pairs")
    resume_val = int(os.getenv("RESUME_VAL", "0"))
    run_generation(df_val, paths["val_jsonl"], generator, resume_idx=resume_val)
    validate_jsonl(paths["val_jsonl"], "VAL")

    banner("Data prep complete ✓")
    log.info(f"  Train JSONL → {paths['train_jsonl']}")
    log.info(f"  Val   JSONL → {paths['val_jsonl']}")
    log.info("Next step: python part2_fine_tuning/run_train.py")


if __name__ == "__main__":
    main()
