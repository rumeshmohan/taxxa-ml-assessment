#!/usr/bin/env python3
"""
Standalone inference snippet — required deliverable for Part 2.

Reproduces the evaluation numbers exactly. Use the sampling params below
(temperature=0.1, repetition_penalty=1.1) to match the eval run.

Usage:
    # Single question
    python part2_fine_tuning/inference.py

    # Or import as a module:
    from part2_fine_tuning.inference import load_model, generate_answer
    model, tok = load_model()
    print(generate_answer(model, tok, "Miten tutkimusmenot käsitellään kirjanpidossa?"))
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Default adapter path — override with ADAPTER_PATH env var or pass directly
_ROOT        = Path(__file__).parent.parent
ADAPTER_DIR  = str(_ROOT / "part2_fine_tuning" / "outputs" / "kila-adapter" / "final_adapter")

# Sampling params used in eval — do not change for reproducibility
GENERATION_CONFIG = dict(
    max_new_tokens=300,
    temperature=0.1,        # low temperature → deterministic, factual answers
    do_sample=True,
    repetition_penalty=1.1, # discourages looping / copy-paste from context
    # eos_token_id is set dynamically at runtime
)


def load_model(
    base_model_id: str = BASE_MODEL_ID,
    adapter_dir: str = ADAPTER_DIR,
):
    """
    Loads the base model in 4-bit NF4 QLoRA and merges the Kila LoRA adapter.

    Returns:
        model     — PeftModel in eval mode on CUDA (or CPU if no GPU)
        tokenizer — matching tokenizer
    """
    import os
    adapter_dir = os.getenv("ADAPTER_PATH", adapter_dir)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model in 4-bit …")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    print("Model ready.\n")
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 300,
) -> str:
    """
    Generates a Finnish accounting answer for the given question.

    Prompt format: Qwen2.5 native ChatML with a Finnish accounting system prompt.

    Args:
        model          — loaded PeftModel
        tokenizer      — matching tokenizer
        question       — Finnish question string
        max_new_tokens — max tokens to generate (default: 300)

    Returns:
        answer string (assistant turn only, special tokens stripped)
    """
    prompt = (
        "<|im_start|>system\n"
        "Olet suomalaisen kirjanpidon asiantuntija. "
        "Vastaa kysymyksiin Kilan ohjeiden pohjalta.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            **{k: v for k, v in GENERATION_CONFIG.items() if k != "max_new_tokens"},
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant turn only
    return decoded.split("<|im_start|>assistant")[-1].strip()


DEMO_QUESTIONS = [
    "Mihin kantaverkon liittymismaksu merkitään taseessa?",
    "Miten kantaverkon liittymismaksu poistetaan?",
    "Mikä on oikea tapa jaksottaa ohjelmistolisenssin käyttöoikeuden tulot?",
    "Miten tutkimusmenot käsitellään kirjanpidossa Kilan ohjeiden mukaan?",
    "Milloin tilinpäätös on laadittava tilikauden päättymisestä?",
]


if __name__ == "__main__":
    model, tokenizer = load_model()

    print("=" * 60)
    print("  Kila Fine-Tuned Model — Demo Inference")
    print("=" * 60)

    for q in DEMO_QUESTIONS:
        print(f"\nQ: {q}")
        ans = generate_answer(model, tokenizer, q)
        print(f"A: {ans}\n")
        print("-" * 60)
