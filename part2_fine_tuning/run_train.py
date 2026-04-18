#!/usr/bin/env python3
"""
Part 2 — Step 2: QLoRA Fine-Tuning
QLoRA fine-tuning of Qwen2.5-7B-Instruct on synthetic Finnish Kila Q&A pairs.

Designed for NVIDIA Blackwell (sm_120): RTX 5090 (32GB) or RTX PRO 6000 (96GB).
Blackwell requirements applied:
  - bf16=True  (not fp16)
  - gradient_checkpointing_kwargs={"use_reentrant": False}
  - optim="paged_adamw_32bit"

Run from project root:
    python part2_fine_tuning/run_train.py

Outputs:
    part2_fine_tuning/outputs/kila-adapter/final_adapter/   — LoRA weights
    part2_fine_tuning/outputs/kila-adapter/train_metrics.json
    part2_fine_tuning/outputs/training_loss_curve.png
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from utils.config import PARAMS, resolve_ft_data_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FT = PARAMS.get("finetuning", {})


def banner(msg: str):
    log.info("=" * 60)
    log.info(f"  {msg}")
    log.info("=" * 60)


def check_environment():
    banner("STEP 2A — Environment check")
    log.info(f"Python  : {sys.version.split()[0]}")
    log.info(f"PyTorch : {torch.__version__}")
    log.info(f"CUDA    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        cap  = torch.cuda.get_device_capability(0)
        log.info(f"GPU     : {name}  ({vram:.0f} GB)")
        log.info(f"Compute : sm_{cap[0]}{cap[1]}")
        if cap[0] >= 12:
            log.info("  ✓ Blackwell detected — bf16 + use_reentrant=False will be used.")
        else:
            log.warning("  Non-Blackwell GPU: settings still work but not optimised.")
    else:
        log.warning("No GPU found — training on CPU will be very slow.")

    import transformers, peft, trl, bitsandbytes
    log.info(f"transformers  : {transformers.__version__}")
    log.info(f"peft          : {peft.__version__}")
    log.info(f"trl           : {trl.__version__}")
    log.info(f"bitsandbytes  : {bitsandbytes.__version__}")


def load_model_and_tokenizer(model_id: str):
    banner("STEP 2B — Loading model in 4-bit (QLoRA)")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    log.info(f"Loading tokenizer from {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info("Loading model in 4-bit … (takes ~2–3 min)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache      = False
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required for sm_120
    )
    log.info("Model loaded and prepared for QLoRA.")
    return model, tokenizer


def apply_lora(model):
    banner("STEP 2C — Applying LoRA")
    from peft import LoraConfig, get_peft_model

    lora_cfg = FT.get("lora", {})
    config   = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"Trainable params : {trainable:,}  ({100*trainable/total:.2f}% of total)")
    log.info(f"Total params     : {total:,}")
    log.info(f"LoRA rank r      : {config.r}  |  alpha: {config.lora_alpha}")
    return model


def prepare_datasets(tokenizer, paths):
    banner("STEP 2D — Preparing datasets")
    from datasets import load_dataset
    import numpy as np

    train_path = paths["train_jsonl"]
    val_path   = paths["val_jsonl"]

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training JSONL not found: {train_path}\n"
            "Run run_data_prep.py first."
        )
    if not val_path.exists():
        log.warning(f"Val JSONL not found: {val_path}. Training without validation.")

    data_files = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)

    raw = load_dataset("json", data_files=data_files)
    log.info(f"Train size : {len(raw['train']):,}")
    if "validation" in raw:
        log.info(f"Val size   : {len(raw['validation']):,}")

    max_seq = FT.get("model", {}).get("max_seq_length", 2048)

    def format_instruction(ex):
        return {
            "text": (
                "<|im_start|>system\n"
                "Olet suomalaisen kirjanpidon asiantuntija. Vastaat kysymyksiin Kilan "
                "(Kirjanpitolautakunta) ohjeiden pohjalta.<|im_end|>\n"
                f"<|im_start|>user\n{ex.get('instruction','')}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex.get('response','')}<|im_end|>"
            )
        }

    train_ds = raw["train"].map(format_instruction, remove_columns=raw["train"].column_names)
    val_ds   = raw["validation"].map(format_instruction, remove_columns=raw["validation"].column_names) \
               if "validation" in raw else None

    sample_lens = [
        len(tokenizer(ex["text"])["input_ids"])
        for ex in train_ds.select(range(min(100, len(train_ds))))
    ]
    log.info(f"Token p50: {int(np.percentile(sample_lens, 50))} | "
             f"p95: {int(np.percentile(sample_lens, 95))} | "
             f"max_seq_length: {max_seq}")
    pct = sum(1 for l in sample_lens if l > max_seq) / len(sample_lens) * 100
    log.info(f"Est. truncated: {pct:.1f}% of sampled examples")

    return train_ds, val_ds


def run_training(model, tokenizer, train_ds, val_ds):
    banner("STEP 2E — Training")
    from trl import SFTConfig, SFTTrainer

    tr      = FT.get("training", {})
    out     = _ROOT / tr.get("output_dir", "part2_fine_tuning/outputs/kila-adapter")
    max_seq = FT.get("model", {}).get("max_seq_length", 2048)

    sft_config = SFTConfig(
        output_dir=str(out),
        max_length=max_seq,
        dataset_text_field="text",
        per_device_train_batch_size=tr.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=tr.get("gradient_accumulation_steps", 4),
        learning_rate=tr.get("learning_rate", 2e-4),
        num_train_epochs=tr.get("num_train_epochs", 3),
        lr_scheduler_type=tr.get("lr_scheduler_type", "cosine"),
        warmup_steps=tr.get("warmup_steps", 10),
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required for sm_120
        optim="paged_adamw_32bit",
        logging_steps=tr.get("logging_steps", 5),
        save_strategy="steps",
        save_steps=tr.get("save_steps", 50),
        save_total_limit=tr.get("save_total_limit", 2),
        eval_strategy="epoch" if val_ds is not None else "no",
        report_to="tensorboard",
        seed=42,
        data_seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    log.info("Starting training …")
    train_result = trainer.train()
    log.info(f"Training complete. Final loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    return trainer, train_result, out


def save_outputs(trainer, train_result, out_dir: Path):
    banner("STEP 2F — Saving adapter & metrics")
    adapter_path = out_dir / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    trainer.tokenizer.save_pretrained(str(adapter_path))

    metrics_path = out_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    log.info(f"Adapter saved → {adapter_path}")
    log.info(f"Metrics saved → {metrics_path}")
    log.info(f"Files: {[p.name for p in adapter_path.iterdir()]}")

    _save_loss_curve(trainer.state.log_history, out_dir)


def _save_loss_curve(log_history, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(log_history)
        plt.figure(figsize=(10, 6))

        if "loss" in df.columns:
            train_df = df[df["loss"].notna()]
            if not train_df.empty:
                plt.plot(train_df["step"], train_df["loss"],
                         label="Train loss", color="steelblue", lw=2)

        if "eval_loss" in df.columns:
            eval_df = df[df["eval_loss"].notna()]
            plt.plot(eval_df["step"], eval_df["eval_loss"],
                     label="Val loss", color="darkorange", marker="o", linestyle="--")

        plt.title("Qwen2.5-7B Kila Fine-tuning — Loss Curve")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        png_path = _ROOT / PARAMS.get("finetuning", {}).get("eval", {}).get(
            "loss_curve_png", "part2_fine_tuning/outputs/training_loss_curve.png"
        )
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(png_path, dpi=150)
        plt.close()
        log.info(f"Loss curve saved → {png_path}")
    except Exception as exc:
        log.warning(f"Could not save loss curve: {exc}")


def main():
    banner("Part 2 — Step 2: QLoRA Fine-Tuning")
    check_environment()

    paths    = resolve_ft_data_paths()
    model_id = FT.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-7B-Instruct")

    model, tokenizer           = load_model_and_tokenizer(model_id)
    model                      = apply_lora(model)
    train_ds, val_ds           = prepare_datasets(tokenizer, paths)
    trainer, train_result, out = run_training(model, tokenizer, train_ds, val_ds)
    save_outputs(trainer, train_result, out)

    banner("Training complete ✓")
    log.info("Next step: python part2_fine_tuning/run_eval.py")


if __name__ == "__main__":
    main()