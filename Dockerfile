# NVIDIA NGC base — CUDA 12.8 + PyTorch 2.6 + cuDNN 9, sm_120 (Blackwell) native
FROM nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://astral.sh/uv/install.sh | sh

# PyTorch is already in the NGC base — install remaining deps on top.
# bitsandbytes must be built from source for sm_120 until an official cu128
# wheel ships; the --no-build-isolation flag lets it see the system CUDA headers.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra finetune --no-build-isolation \
    --reinstall-package bitsandbytes

COPY . .

# Part 1 (RAG eval — CPU only, no GPU needed):
#   docker run -e GROQ_API_KEY=xxx -v /host/data:/data \
#     -v /host/p1_out:/app/part1_rag_eval/outputs \
#     taxxa uv run python part1_rag_eval/run.py
#
# Part 2 — training (GPU required):
#   docker run --gpus all -v /host/data:/data \
#     -v /host/p2_out:/app/part2_fine_tuning/outputs \
#     taxxa uv run python part2_fine_tuning/run_train.py
#
# Part 2 — eval:
#   docker run --gpus all -e GROQ_API_KEY=xxx -v /host/data:/data \
#     -v /host/p2_out:/app/part2_fine_tuning/outputs \
#     taxxa uv run python part2_fine_tuning/run_eval.py

CMD ["uv", "run", "python", "part2_fine_tuning/run_train.py"]