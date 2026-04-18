#!/usr/bin/env python3
"""
Part 2 — Step 3: Evaluation
Step 3 of Part 2: Evaluate base model vs fine-tuned adapter on the 12
official Finnish eval questions.

Deliverables (all written to outputs/):
  - eval_summary.csv         — correctness / grounding / fluency scores
  - eval_answers.csv         — per-question answers from both models
  - eval_failure_cases.csv   — questions where FT did not improve
  - training_loss_curve.png  — (already saved by run_train.py)

Judge: Groq llama-3.3-70b-versatile (configurable via params.yaml)
Rubric documented inline — required by the assessment.

Run from project root:
    python part2_fine_tuning/run_eval.py

Env vars:
    DATA_DIR          path to parquet data (default: ./data)
    GROQ_API_KEY      required (judge uses Groq 70B by default)
    ADAPTER_PATH      override the adapter directory
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from utils.config       import PARAMS, resolve_ft_data_paths, get_active_generator_config
from utils.llm_services import LLMService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FT  = PARAMS.get("finetuning", {})
EV  = FT.get("eval", {})


def banner(msg: str):
    log.info("=" * 60)
    log.info(f"  {msg}")
    log.info("=" * 60)


def load_eval_questions(paths) -> List[str]:
    banner("STEP 3A — Loading Finnish eval questions")
    q_path = paths["eval_queries_path"]

    if not q_path.exists():
        raise FileNotFoundError(
            f"Eval queries not found: {q_path}\n"
            "Set DATA_DIR or check params.yaml → finetuning.data.eval_queries_file"
        )
    df = pd.read_parquet(q_path)
    log.info(f"Loaded {len(df)} questions | columns: {df.columns.tolist()}")

    # Try common column names for the question text
    for col in ["question_text", "query_text", "question", "query"]:
        if col in df.columns:
            questions = df[col].tolist()
            log.info(f"Using column '{col}' for question text.")
            return questions, df

    raise ValueError(
        f"Cannot find question text column. Available: {df.columns.tolist()}"
    )


def load_base_model(model_id: str):
    banner("STEP 3B — Loading base model (4-bit)")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    log.info(f"Base model loaded: {model_id}")
    return model, tokenizer


def load_finetuned_model(base_model, adapter_path: str):
    banner("STEP 3B — Loading fine-tuned adapter")
    from peft import PeftModel

    if not Path(adapter_path).exists():
        raise FileNotFoundError(
            f"Adapter not found: {adapter_path}\n"
            "Run run_train.py first, or set ADAPTER_PATH env var."
        )
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()
    log.info(f"Adapter loaded from: {adapter_path}")
    return ft_model


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 300) -> str:
    """
    Generates a Finnish accounting answer using the Qwen2.5 ChatML prompt format.
    Sampling params fixed for reproducibility (temp=0.1, no nucleus sampling).
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
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant turn
    return full.split("<|im_start|>assistant")[-1].strip()


def batch_generate(model, tokenizer, questions: List[str], label: str) -> List[str]:
    log.info(f"Generating answers ({label}) for {len(questions)} questions …")
    return [
        generate_answer(model, tokenizer, q)
        for q in tqdm(questions, desc=label)
    ]


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of Finnish accounting answers.
Score the provided answer on three dimensions, each from 1 to 5:

- correctness : Is the factual content accurate according to Finnish accounting
                standards (Kila / kirjanpitolaki)? 1 = completely wrong, 5 = fully correct.
- grounding   : Does the answer reference or stay consistent with authoritative sources
                (Kila statements, legislation)? 1 = no grounding, 5 = well-grounded.
- fluency     : Is the Finnish grammatically correct and professionally written?
                1 = broken / incomprehensible, 5 = fluent professional Finnish.

Return ONLY a JSON object with keys: correctness, grounding, fluency, reason.
Example: {"correctness": 4, "grounding": 3, "fluency": 5, "reason": "..."}"""


def judge_answer(
    llm: LLMService,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Scores one answer using the LLM-as-judge rubric above.
    Returns dict with correctness/grounding/fluency/reason keys.
    """
    prompt = (
        f"Question (Finnish): {question}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        "Score this answer according to the rubric. Return JSON only."
    )
    result = llm.generate_json(prompt, system_prompt=JUDGE_SYSTEM_PROMPT, temperature=0.0)

    # Validate structure
    if isinstance(result, dict) and "correctness" in result:
        # Clamp scores to [1, 5]
        for k in ("correctness", "grounding", "fluency"):
            if k in result:
                result[k] = max(1, min(5, int(result[k])))
        return result

    log.warning(f"Judge returned unexpected format: {result}")
    return {"correctness": None, "grounding": None, "fluency": None, "reason": str(result)}


def batch_judge(
    llm: LLMService,
    questions: List[str],
    answers: List[str],
    label: str,
) -> List[Dict[str, Any]]:
    log.info(f"Judging {len(questions)} answers ({label}) …")
    scores = []
    for q, a in tqdm(zip(questions, answers), total=len(questions), desc=f"Judge {label}"):
        scores.append(judge_answer(llm, q, a))
        time.sleep(2.1)  # Groq 30 RPM pacing
    return scores


def compile_results(
    questions: List[str],
    base_answers: List[str],
    ft_answers: List[str],
    base_scores: List[Dict],
    ft_scores: List[Dict],
    out_dir: Path,
):
    banner("STEP 3E — Compiling results")
    metrics = ["correctness", "grounding", "fluency"]

    def safe_mean(scores_list, key):
        vals = [s[key] for s in scores_list if isinstance(s.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    summary_rows = []
    for m in metrics:
        b = safe_mean(base_scores, m)
        f = safe_mean(ft_scores,   m)
        delta = round(f - b, 2) if (b is not None and f is not None) else None
        summary_rows.append({"Metric": m.capitalize(), "Base": b, "Fine-Tuned": f, "Delta": delta})

    df_summary = pd.DataFrame(summary_rows)
    log.info(f"\n{df_summary.to_string(index=False)}")

    df_answers = pd.DataFrame({
        "question":        questions,
        "base_answer":     base_answers,
        "ft_answer":       ft_answers,
        "base_correctness":[s.get("correctness") for s in base_scores],
        "ft_correctness":  [s.get("correctness") for s in ft_scores],
        "base_grounding":  [s.get("grounding")   for s in base_scores],
        "ft_grounding":    [s.get("grounding")    for s in ft_scores],
        "base_fluency":    [s.get("fluency")      for s in base_scores],
        "ft_fluency":      [s.get("fluency")      for s in ft_scores],
        "ft_judge_reason": [s.get("reason", "")   for s in ft_scores],
    })

    # Failure cases (FT did not improve correctness)
    df_failures = df_answers[
        df_answers["ft_correctness"] <= df_answers["base_correctness"]
    ].copy()
    log.info(f"Failure / regression cases: {len(df_failures)} / {len(questions)}")

    # Manual spot-check: top-5 most interesting (largest delta)
    df_answers["delta"] = df_answers["ft_correctness"] - df_answers["base_correctness"]
    df_spot = df_answers.nlargest(5, "delta")[
        ["question", "base_answer", "ft_answer",
         "base_correctness", "ft_correctness", "delta", "ft_judge_reason"]
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_dir / "eval_summary.csv",         index=False)
    df_answers.to_csv(out_dir / "eval_answers.csv",         index=False)
    df_failures.to_csv(out_dir / "eval_failure_cases.csv",  index=False)
    df_spot.to_csv(out_dir / "eval_spot_check.csv",         index=False)

    log.info(f"Results saved → {out_dir}")

    banner("Manual spot-check (top-5 Δcorrectness)")
    for _, row in df_spot.iterrows():
        log.info(f"\nQ:    {row['question']}")
        log.info(f"Base ({row['base_correctness']}): {str(row['base_answer'])[:200]}")
        log.info(f"FT   ({row['ft_correctness']}): {str(row['ft_answer'])[:200]}")
        log.info(f"Δ: {row['delta']}  | Reason: {str(row['ft_judge_reason'])[:150]}")

    return df_summary, df_answers


def main():
    banner("Part 2 — Step 3: Evaluation")

    paths     = resolve_ft_data_paths()
    questions, df_eval = load_eval_questions(paths)

    model_id     = FT.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-7B-Instruct")
    adapter_path = os.getenv(
        "ADAPTER_PATH",
        str(_ROOT / EV.get("adapter_path", "part2_fine_tuning/outputs/kila-adapter/final_adapter"))
    )

    base_model, tokenizer = load_base_model(model_id)
    ft_model              = load_finetuned_model(base_model, adapter_path)

    banner("STEP 3C — Generating answers")
    base_answers = batch_generate(base_model, tokenizer, questions, label="Base")
    ft_answers   = batch_generate(ft_model,   tokenizer, questions, label="Fine-Tuned")

    banner("STEP 3D — LLM-as-judge scoring")
    judge_provider = EV.get("judge_provider", "groq")
    judge_model    = EV.get("judge_model", "llama-3.3-70b-versatile")
    judge_api_key  = os.getenv(f"{judge_provider.upper()}_API_KEY", "")

    judge_bases = {
        "groq":   "https://api.groq.com/openai/v1",
        "openai": "https://api.openai.com/v1",
    }
    judge_llm = LLMService(
        base_url=judge_bases.get(judge_provider, "https://api.groq.com/openai/v1"),
        api_key=judge_api_key,
        model_name=judge_model,
    )
    log.info(f"Judge: {judge_provider.upper()} / {judge_model}")
    log.info("Rubric: correctness(1-5) | grounding(1-5) | fluency(1-5)")

    base_scores = batch_judge(judge_llm, questions, base_answers, "Base")
    ft_scores   = batch_judge(judge_llm, questions, ft_answers,   "Fine-Tuned")

    out_dir = _ROOT / "part2_fine_tuning" / "outputs"
    compile_results(questions, base_answers, ft_answers, base_scores, ft_scores, out_dir)

    banner("Evaluation complete ✓")
    log.info(f"All outputs → {out_dir}")


if __name__ == "__main__":
    main()
