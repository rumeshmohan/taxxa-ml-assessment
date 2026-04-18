# taxxa_assessment/Dockerfile
# Supports both Part 1 (RAG eval) and Part 2 (fine-tuning).
# Base image includes CUDA 12.4 for Blackwell (sm_120) compatibility.

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Python deps — Part 1 only by default
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# App code (data is volume-mounted, not baked in)
COPY . .

# Default env
ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

# ── Usage examples ────────────────────────────────────────────────────────────
# Part 1 (RAG eval):
#   docker run -e DATA_DIR=/data -e GROQ_API_KEY=xxx \
#     -v /host/data:/data -v /host/outputs:/app/part1_rag_eval/outputs \
#     taxxa uv run python part1_rag_eval/run.py
#
# Part 2 — data prep:
#   docker run -e DATA_DIR=/data -e GROQ_API_KEY=xxx \
#     -v /host/data:/data taxxa uv run python part2_fine_tuning/run_data_prep.py
#
# Part 2 — training (requires GPU):
#   docker run --gpus all -e DATA_DIR=/data \
#     -v /host/data:/data -v /host/outputs:/app/part2_fine_tuning/outputs \
#     taxxa uv run python part2_fine_tuning/run_train.py
#
# Part 2 — eval:
#   docker run --gpus all -e DATA_DIR=/data -e GROQ_API_KEY=xxx \
#     -v /host/data:/data -v /host/outputs:/app/part2_fine_tuning/outputs \
#     taxxa uv run python part2_fine_tuning/run_eval.py

CMD ["uv", "run", "python", "part1_rag_eval/run.py"]