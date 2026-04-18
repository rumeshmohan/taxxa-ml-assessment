# taxxa_assessment/utils/config.py
import os
import yaml
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Root = taxxa_assessment/ (one level above utils/)
ROOT_DIR = Path(__file__).parent.parent.absolute()

# DATA_DIR: override with env var when running in a pod with a mounted volume
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))

load_dotenv(dotenv_path=ROOT_DIR / ".env")


def load_yaml(file_name: str) -> Dict[str, Any]:
    path = ROOT_DIR / "config" / file_name
    if not path.exists():
        print(f"[WARN] {file_name} not found at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


MODELS: Dict[str, Any] = load_yaml("models.yaml")
PARAMS: Dict[str, Any] = load_yaml("params.yaml")


def get_active_generator_config() -> Dict[str, str]:
    """Resolves active LLM provider/model/key from params + env."""
    provider = PARAMS.get("active_generator", {}).get("provider", "ollama")
    tier     = PARAMS.get("active_generator", {}).get("tier", "general")
    model    = MODELS.get(provider, {}).get("chat", {}).get(tier, "")
    api_key  = os.getenv(f"{provider.upper()}_API_KEY", "dummy_key")

    defaults = {
        "openai":     "https://api.openai.com/v1",
        "groq":       "https://api.groq.com/openai/v1",
        "deepseek":   "https://api.deepseek.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai/",
        "mistral":    "https://api.mistral.ai/v1",
        "ollama":     "http://localhost:11434/v1",
    }
    base_url = os.getenv(f"{provider.upper()}_BASE_URL", defaults.get(provider, ""))

    return {"provider": provider, "model_name": model, "api_key": api_key, "base_url": base_url}


def resolve_rag_data_paths() -> Dict[str, Any]:
    """Resolves Part 1 data paths relative to DATA_DIR."""
    cfg = PARAMS.get("rag", {}).get("data", {})
    return {
        "source_dirs":  [DATA_DIR / s for s in cfg.get("sources", [])],
        "queries_path": DATA_DIR / cfg.get("queries_file", "queries_dk.parquet"),
        "max_rets":     cfg.get("max_retsinformation_parts", 10),
    }


def resolve_ft_data_paths() -> Dict[str, Any]:
    """Resolves Part 2 data paths relative to DATA_DIR and ROOT_DIR."""
    cfg = PARAMS.get("finetuning", {}).get("data", {})
    return {
        "chunk_dirs":         [DATA_DIR / s for s in cfg.get("chunks_sources", [])],
        "eval_queries_path":  DATA_DIR / cfg.get("eval_queries_file", "queries_fi.parquet"),
        "train_jsonl":        ROOT_DIR / cfg.get("synthetic_train_path", "part2_fine_tuning/data/synthetic_kila_train.jsonl"),
        "val_jsonl":          ROOT_DIR / cfg.get("synthetic_val_path",   "part2_fine_tuning/data/synthetic_kila_val.jsonl"),
        "val_doc_ids_path":   ROOT_DIR / cfg.get("val_doc_ids_path",     "part2_fine_tuning/data/val_doc_ids.json"),
    }
