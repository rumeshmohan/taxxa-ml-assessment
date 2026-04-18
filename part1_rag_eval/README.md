# Part 1 — RAG Evaluation on Denmark KB

Reproducible evaluation harness comparing three retrieval strategies against the Denmark Knowledge Base (Retsinformation, Erhvervsstyrelsen, Skat.dk).

---

## Quick start

```bash
git clone https://github.com/rumeshmohan/taxxa-ml-assessment.git
cd taxxa-ml-assessment

uv sync                       # installs Part 1 dependencies
cp .env.example .env          # fill in GROQ_API_KEY (only needed for generation)
export DATA_DIR=/path/to/data # root folder containing the parquet files
```

> **PyTorch / CUDA**: Install torch separately to match your CUDA version:
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```

---

## Run

```bash
# Retrieval metrics only (no LLM or Ollama required — this is the submitted run)
SKIP_GENERATION=1 SKIP_RAGAS=1 python part1_rag_eval/run.py

# Full run (retrieval + generation + RAGAS — requires Ollama running locally)
python part1_rag_eval/run.py
```

**Output:** `part1_rag_eval/outputs/part1_results.csv`

---

## Data layout

Place the Taxxa-provided SAS-downloaded files under `DATA_DIR`:

```
DATA_DIR/
├── skatdk/2026-04-09/
│   ├── documents.parquet
│   ├── chunks_part-*.parquet
│   └── embeddings_part-*.parquet
├── erhvervsstyrelsen/2026-04-09/
│   └── ...
├── retsinformation/2026-04-09/
│   └── ...                        ← first 10 part-files loaded (OOM guard)
└── queries_dk_2026-04-09.parquet  ← evaluation queries
```

Source paths and date slugs are configured in `config/params.yaml`.

---

## Retrieval strategies

| Strategy | Description |
|----------|-------------|
| **Dense** | FAISS `IndexFlatIP` over pre-computed `text-embedding-3-large` (3072-dim) vectors. Inner product = cosine similarity for normalised embeddings. |
| **Hybrid (RRF)** | Reciprocal Rank Fusion (`k=60`) of Dense + `BM25Okapi`. Balances semantic understanding with exact keyword/ID matching for tax forms and accounting codes. No re-embedding needed. |
| **Graph** | `MetadataGraphRetriever`: runs Hybrid, takes the top-1 result, expands via `breadcrumbs` prefix siblings (depth-2), and re-ranks with a `0.85x` score penalty. Rationale: Danish legal content is deeply hierarchical — pulling regulatory siblings improves recall on clause-level questions. |

---

## Metrics reported

| Metric | Notes |
|--------|-------|
| Recall@1, @5, @10 | Strict `chunk_id` match against ground-truth |
| nDCG@10 | Graded relevance ranking |
| Latency p50, p95 | Wall-clock per query |
| Cost per 1k queries | Retrieval only: ~$0.0026 |
| Faithfulness, Answer Relevancy | RAGAS via local Ollama `llama3.1:8b` — only available on full run |

**Judge model:** Not applicable — this submission is a retrieval-only run (`SKIP_GENERATION=1 SKIP_RAGAS=1`). Faithfulness and Answer Relevancy are not reported. To run the full pipeline with RAGAS scoring, set the active provider in `config/params.yaml` (`active_generator.provider`) and supply the corresponding API key in `.env` — Groq, OpenAI, DeepSeek, and local Ollama are all supported.

---

## Configuration

All tuneable parameters live in `config/params.yaml`. Key Part 1 knobs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_retsinformation_parts` | `10` | Part-file cap for Retsinformation (OOM guard) |
| `rrf_k` | `60` | RRF fusion constant |
| `graph_sibling_penalty` | `0.85` | Score multiplier for injected breadcrumb siblings |
| `top_k` | `10` | Number of chunks retrieved per query |

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATA_DIR` | Yes | Root folder containing all parquet data |
| `GROQ_API_KEY` | No* | Only needed if running generation (`SKIP_GENERATION` not set) |
| `SKIP_GENERATION` | No | Set to `1` to skip LLM answer generation |
| `SKIP_RAGAS` | No | Set to `1` to skip Ollama RAGAS scoring |

---

## Docker (optional)

```bash
docker build -t taxxa .

# Retrieval-only run
docker run \
  -e DATA_DIR=/data \
  -e SKIP_GENERATION=1 \
  -e SKIP_RAGAS=1 \
  -v /host/data:/data \
  -v /host/outputs:/app/part1_rag_eval/outputs \
  taxxa python part1_rag_eval/run.py
```