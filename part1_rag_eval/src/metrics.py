# part1_rag_eval/src/metrics.py
import math
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: List[str], ground_truth: List[str], k: int) -> float:
    hits = sum(1 for gid in ground_truth if gid in retrieved[:k])
    return hits / len(ground_truth) if ground_truth else 0.0


def ndcg_at_k(retrieved: List[str], ground_truth: List[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(retrieved[:k])
        if rid in ground_truth
    )
    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(ground_truth), k))
    )
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Stateful tracker (one per retriever strategy)
# ---------------------------------------------------------------------------

class EvaluationTracker:
    def __init__(self):
        self.latencies:  List[float] = []
        self.recalls_1:  List[float] = []
        self.recalls_5:  List[float] = []
        self.recalls_10: List[float] = []
        self.ndcgs_10:   List[float] = []

    def record_query(
        self,
        retrieved_ids: List[str],
        ground_truth_ids: List[str],
        duration_sec: float,
    ):
        self.latencies.append(duration_sec)
        self.recalls_1.append(recall_at_k(retrieved_ids, ground_truth_ids, 1))
        self.recalls_5.append(recall_at_k(retrieved_ids, ground_truth_ids, 5))
        self.recalls_10.append(recall_at_k(retrieved_ids, ground_truth_ids, 10))
        self.ndcgs_10.append(ndcg_at_k(retrieved_ids, ground_truth_ids, 10))

    def get_summary(self, cost_per_query_usd: float = 0.0) -> Dict[str, Any]:
        if not self.latencies:
            return {}
        return {
            "Recall@1":               round(float(np.mean(self.recalls_1)),  4),
            "Recall@5":               round(float(np.mean(self.recalls_5)),  4),
            "Recall@10":              round(float(np.mean(self.recalls_10)), 4),
            "nDCG@10":                round(float(np.mean(self.ndcgs_10)),   4),
            "p50_latency_sec":        round(float(np.percentile(self.latencies, 50)), 4),
            "p95_latency_sec":        round(float(np.percentile(self.latencies, 95)), 4),
            "avg_latency_sec":        round(float(np.mean(self.latencies)), 4),
            "cost_per_1k_queries_usd": round(cost_per_query_usd * 1000, 4),
        }


# ---------------------------------------------------------------------------
# RAGAS faithfulness + answer relevancy (local Ollama judge)
# ---------------------------------------------------------------------------

def evaluate_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ollama_model: str = "llama3.1:8b",
    ollama_embed_model: str = "nomic-embed-text",
) -> Dict[str, float]:
    """
    Runs RAGAS faithfulness + answer_relevancy using a local Ollama judge.
    Zero cost, no external API key required.

    Judge bias note: local 8B models tend to over-score faithfulness when the
    answer closely paraphrases the context. Treat absolute numbers as relative
    indicators, not ground truth.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    try:
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
        judge_llm    = ChatOllama(model=ollama_model, temperature=0.0)
        judge_embeds = OllamaEmbeddings(model=ollama_embed_model)
    except ImportError as exc:
        raise ImportError(
            "langchain-community is required for RAGAS. "
            "Run: pip install langchain-community"
        ) from exc

    dataset = Dataset.from_dict({
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    })

    print(f"[RAGAS] Judge model: {ollama_model}")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeds,
    )
    return dict(result)
