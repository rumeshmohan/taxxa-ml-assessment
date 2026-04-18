# Taxxa ML Assessment — RAG Evaluation & Fine-Tuning

Two-part submission for the Taxxa ML Engineer take-home assessment.

| Part | Goal | Entry point | Write-up |
|------|------|-------------|----------|
| 1 | RAG evaluation on Denmark KB (Dense / Hybrid / Graph) | `uv run python part1_rag_eval/run.py` | [`part1_rag_eval/writeup.md`](part1_rag_eval/writeup.md) |
| 2 | QLoRA fine-tuning of Qwen2.5-7B on Finnish Kila corpus | `uv run python part2_fine_tuning/run_*.py` | [`part2_fine_tuning/writeup.md`](part2_fine_tuning/writeup.md) |

---

## Quick start

```bash
git clone https://github.com/rumeshmohan/taxxa-ml-assessment.git
cd taxxa-ml-assessment

uv sync                       # Part 1 dependencies
uv sync --extra finetune      # Part 1 + Part 2 dependencies
cp .env.example .env          # fill in GROQ_API_KEY at minimum
export DATA_DIR=/path/to/data # parquet files root
```

> **PyTorch / CUDA (local / non-Docker)**: Install torch separately to match your CUDA version:
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```
> When using Docker (SimplePod), this is handled automatically by the NGC base image — no manual torch install needed.

---

## Data layout under `DATA_DIR`

```
DATA_DIR/
├── skatdk/2026-04-09/
│   ├── documents.parquet
│   ├── chunks_part-*.parquet
│   └── embeddings_part-*.parquet
├── erhvervsstyrelsen/2026-04-09/
│   └── ...
├── retsinformation/2026-04-09/
│   └── ...                        ← first 10 parts loaded (OOM guard)
├── kila/2026-04-09/
│   ├── documents.parquet
│   └── chunks_part-*.parquet
├── queries_dk_2026-04-09.parquet  ← Part 1 eval queries
└── queries_fi_2026-04-09.parquet  ← Part 2 eval queries
```

Source paths and filenames are configured in `config/params.yaml`.

---

## Part 1 — RAG Evaluation

```bash
# Retrieval metrics only (this is the submitted run)
SKIP_GENERATION=1 SKIP_RAGAS=1 uv run python part1_rag_eval/run.py

# Full run (retrieval + generation + RAGAS)
uv run python part1_rag_eval/run.py
```

**Output:** `part1_rag_eval/outputs/part1_results.csv`

| Strategy | Recall@1 | Recall@5 | Recall@10 | nDCG@10 | p50_latency_sec | cost_per_1k_queries_usd |
|----------|----------|----------|-----------|---------|-----------------|--------------------------|
| Dense    | 0.0      | 0.0      | 0.0       | 0.0     | 0.71s           | ~$0.0026                 |
| Hybrid   | 0.0      | 0.0      | 0.0       | 0.0     | 0.76s           | ~$0.0026                 |
| Graph    | 0.0      | 0.0      | 0.0       | 0.0     | 0.76s           | ~$0.0026                 |

> Retrieval-only run. Faithfulness and Answer Relevancy not reported. See `part1_rag_eval/writeup.md` for root cause analysis of 0.0 scores.

**Retrieval strategies:**

- **Dense** — FAISS `IndexFlatIP` over pre-computed `text-embedding-3-large` (3072-dim) vectors. Inner product = cosine similarity for normalised embeddings.
- **Hybrid (RRF)** — Reciprocal Rank Fusion (`k=60`) of Dense + `BM25Okapi`. No re-embedding needed.
- **Graph** — `MetadataGraphRetriever`: runs Hybrid, takes the top-1 result, expands via breadcrumb prefix siblings (depth-2), and re-ranks. Rationale: Danish legal content is deeply hierarchical; pulling regulatory siblings improves recall on clause-level questions.

**RAGAS judge:** Configurable via `active_generator.provider` in `config/params.yaml`. Groq, OpenAI, DeepSeek, and local Ollama are all supported. Set `SKIP_RAGAS=1` to skip scoring entirely.

---

## Part 2 — Fine-Tuning

**Model artifact:** [`rumeshmohan/kila-qwen2.5-7b-adapter`](https://huggingface.co/rumeshmohan/kila-qwen2.5-7b-adapter) *(private — request access)*

**Eval results** (`part2_fine_tuning/outputs/eval_summary.csv`):

| Metric | Base | Fine-Tuned | Delta |
|--------|------|------------|-------|
| Correctness | 4.08 | 3.82 | -0.27 |
| Grounding | 4.33 | 3.91 | -0.42 |
| Fluency | 4.50 | 4.45 | -0.05 |

> Fine-tuned model regressed slightly. See `part2_fine_tuning/writeup.md` for honest failure analysis.

Three sequential steps:

### Step 1 — Data prep

```bash
uv run python part2_fine_tuning/run_data_prep.py

# Resume after a crash (e.g. at chunk 4393):
RESUME_TRAIN=4393 uv run python part2_fine_tuning/run_data_prep.py

# Sanity check on 3 chunks only:
DRY_RUN=1 uv run python part2_fine_tuning/run_data_prep.py
```

Outputs: `part2_fine_tuning/data/synthetic_kila_train.jsonl` + `…_val.jsonl`

### Step 2 — QLoRA training *(requires GPU)*

```bash
uv run python part2_fine_tuning/run_train.py
```

Outputs:
- `part2_fine_tuning/outputs/kila-adapter/final_adapter/` — LoRA weights
- `part2_fine_tuning/outputs/kila-adapter/train_metrics.json`
- `part2_fine_tuning/outputs/training_loss_curve.png`

**Blackwell (sm_120) settings applied:**
- `bf16=True` (not fp16)
- `gradient_checkpointing_kwargs={"use_reentrant": False}`
- `optim="paged_adamw_32bit"`
- `eval_strategy="epoch"` (runs validation each epoch when val set is present)

### Step 3 — Evaluation

```bash
uv run python part2_fine_tuning/run_eval.py
```

Output: `part2_fine_tuning/outputs/eval_summary.csv`

**Judge:** Groq `llama-3.3-70b-versatile` (temp=0.0)  
**Rubric (1–5):** `correctness` — factual accuracy per Kila / kirjanpitolaki | `grounding` — references authoritative sources | `fluency` — professional Finnish grammar

### Inference

```bash
uv run python part2_fine_tuning/inference.py
```

Or as a module:

```python
from part2_fine_tuning.inference import load_model, generate_answer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "./kila-adapter-output/final_adapter"

model, tok = load_model(MODEL_ID, ADAPTER_DIR)
print(generate_answer(model, tok, "Miten tutkimusmenot käsitellään kirjanpidossa?"))
```

---

## Configuration

All tunable parameters: `config/params.yaml`  
Model name ↔ provider mappings: `config/models.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `active_generator.provider` | `groq` | LLM provider for generation + data prep |
| `rag.top_k` | `10` | Retrieved chunks per query |
| `rag.rrf_k` | `60` | RRF fusion constant |
| `rag.embedding_dim` | `3072` | text-embedding-3-large dimension |
| `finetuning.model.base_model_id` | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `finetuning.lora.r` | `16` | LoRA rank |
| `finetuning.training.num_train_epochs` | `3` | Training epochs |

---

## Docker

Base image: `nvcr.io/nvidia/pytorch:25.03-py3` (CUDA 12.8 + PyTorch 2.6 + cuDNN 9, Blackwell sm_120 native).

```bash
docker build -t taxxa .

# Part 1 (CPU only — no --gpus needed)
docker run -e DATA_DIR=/data -e GROQ_API_KEY=$GROQ_API_KEY \
  -v /host/data:/data \
  -v /host/p1_out:/app/part1_rag_eval/outputs \
  taxxa uv run python part1_rag_eval/run.py

# Part 2 — training (GPU required)
docker run --gpus all -e DATA_DIR=/data \
  -v /host/data:/data \
  -v /host/p2_out:/app/part2_fine_tuning/outputs \
  taxxa uv run python part2_fine_tuning/run_train.py

# Part 2 — eval
docker run --gpus all -e DATA_DIR=/data -e GROQ_API_KEY=$GROQ_API_KEY \
  -v /host/data:/data \
  -v /host/p2_out:/app/part2_fine_tuning/outputs \
  taxxa uv run python part2_fine_tuning/run_eval.py
```

> **SimplePod (no Docker):** The pod already has the right CUDA environment. Just run `uv sync --extra finetune --no-build-isolation` and execute scripts directly. Shut the pod down between steps to avoid idle billing.

---

## Project structure

```
taxxa-ml-assessment/
├── config/
│   ├── models.yaml          provider → model name mappings
│   └── params.yaml          all tunable params (both parts)
├── utils/
│   ├── config.py            YAML loading, path resolution, DATA_DIR
│   └── llm_services.py      OpenAI-compatible LLM wrapper
├── part1_rag_eval/
│   ├── run.py               ← entrypoint
│   ├── writeup.md           ← methodology, results, failure analysis
│   ├── src/
│   │   ├── data_loader.py   safe_loader(), preprocess_embeddings()
│   │   ├── retrievers.py    Dense, BM25, Hybrid (RRF)
│   │   ├── graph_utils.py   MetadataGraphRetriever (breadcrumbs)
│   │   └── metrics.py       Recall@k, nDCG@k, EvaluationTracker, RAGAS
│   └── outputs/             results CSV + log (written at runtime)
├── part2_fine_tuning/
│   ├── run_data_prep.py     ← Step 1: load → dedupe → split → generate JSONL
│   ├── run_train.py         ← Step 2: QLoRA fine-tuning
│   ├── run_eval.py          ← Step 3: base vs FT evaluation
│   ├── inference.py         ← inference snippet
│   ├── writeup.md           ← methodology, results, failure analysis
│   ├── src/
│   │   └── dataset_generator.py  ModelAgnosticDataGenerator
│   ├── data/
│   │   ├── synthetic_kila_train.jsonl
│   │   ├── synthetic_kila_val.jsonl
│   │   └── val_doc_ids.json
│   └── outputs/
│       ├── eval_summary.csv
│       └── training_loss_curve.png
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .env.example
└── README.md
```