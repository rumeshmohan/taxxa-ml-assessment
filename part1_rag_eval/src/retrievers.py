import re
from typing import List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


class DenseRetriever:
    """
    FAISS IndexFlatIP over pre-computed text-embedding-3-large vectors.
    Inner Product == Cosine Similarity for L2-normalised embeddings.
    """

    def __init__(self, vectors: np.ndarray, chunk_ids: List[str], dimension: int = 3072):
        self.chunk_ids = chunk_ids
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if query_vector.ndim == 1:
            query_vector = query_vector[np.newaxis, :]
        distances, indices = self.index.search(query_vector, top_k)
        return [(self.chunk_ids[idx], float(dist)) for dist, idx in zip(distances[0], indices[0])]


class BM25Retriever:
    """BM25Okapi over tokenised chunk text."""

    def __init__(self, corpus_texts: List[str], chunk_ids: List[str]):
        self.chunk_ids = chunk_ids
        tokenized = [self._tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        return re.findall(r"\w+", text.lower())

    def search(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        tokens = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx]


class HybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) over Dense + BM25 results.
    rrf_k = 60 is the standard default (from the original RRF paper).
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.dense   = dense_retriever
        self.bm25    = bm25_retriever
        self.rrf_k   = rrf_k

    def search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        candidate_k = max(top_k * 2, 50)
        dense_res = self.dense.search(query_vector, top_k=candidate_k)
        bm25_res  = self.bm25.search(query_text,   top_k=candidate_k)

        rrf: dict = {}
        for results in (dense_res, bm25_res):
            for rank, (chunk_id, _) in enumerate(results, start=1):
                rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
