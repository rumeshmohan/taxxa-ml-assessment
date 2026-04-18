# part1_rag_eval/src/graph_utils.py
from typing import Dict, List, Tuple

import pandas as pd


class MetadataGraphRetriever:
    """
    Graph retriever that exploits the breadcrumb hierarchy baked into the corpus.

    Rationale: Danish legal/tax content is deeply hierarchical.
    A question about "VAT on software" benefits from pulling sibling chunks
    from the same regulatory sub-tree, not just semantically-similar chunks
    spread across the whole corpus.

    Strategy:
      1. Run the base retriever (Hybrid) to get the initial top-k.
      2. Take the top-1 result and find its breadcrumb siblings (same depth-2 prefix).
      3. Inject up to `expand_k` novel siblings with a small score penalty (×0.85).
      4. Re-sort and return top_k.
    """

    def __init__(self, base_retriever, df_corpus: pd.DataFrame):
        self.base_retriever = base_retriever

        def _parse(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                return [b.strip() for b in val.split("|") if b.strip()]
            return []

        df = df_corpus.copy()
        df["_bc"] = df["breadcrumbs"].apply(_parse)

        # chunk_id → breadcrumbs
        self.chunk_to_bc: Dict[str, List[str]] = dict(zip(df["chunk_id"], df["_bc"]))

        # depth-2 prefix → list of chunk_ids sharing that sub-tree
        self.prefix_to_chunks: Dict[tuple, List[str]] = {}
        for _, row in df.iterrows():
            bc = row["_bc"]
            if len(bc) >= 1:
                key = tuple(bc[:2]) if len(bc) >= 2 else (bc[0],)
                self.prefix_to_chunks.setdefault(key, []).append(row["chunk_id"])

        # Fallback: l1_category grouping when breadcrumbs are absent
        self.cat_to_chunks: Dict[str, List[str]] = (
            df.groupby("l1_category")["chunk_id"].apply(list).to_dict()
            if "l1_category" in df.columns else {}
        )
        self.chunk_to_cat: Dict[str, str] = (
            dict(zip(df["chunk_id"], df["l1_category"]))
            if "l1_category" in df.columns else {}
        )

    def _get_neighbors(self, chunk_id: str) -> List[str]:
        bc = self.chunk_to_bc.get(chunk_id, [])
        if bc:
            key = tuple(bc[:2]) if len(bc) >= 2 else (bc[0],)
            return self.prefix_to_chunks.get(key, [])
        cat = self.chunk_to_cat.get(chunk_id)
        return self.cat_to_chunks.get(cat, []) if cat else []

    def search(
        self,
        query_vector,
        query_text: str = "",
        top_k: int = 10,
        expand_k: int = 3,
    ) -> List[Tuple[str, float]]:
        # Base retrieval
        if query_text:
            initial = self.base_retriever.search(query_vector, query_text, top_k=top_k)
        else:
            initial = self.base_retriever.search(query_vector, top_k=top_k)

        if not initial:
            return []

        top_id, top_score = initial[0]
        neighbors = self._get_neighbors(top_id)

        results     = list(initial)
        seen_ids    = {cid for cid, _ in results}
        added       = 0

        for neighbor_id in neighbors:
            if neighbor_id not in seen_ids:
                results.append((neighbor_id, top_score * 0.85))
                seen_ids.add(neighbor_id)
                added += 1
            if added >= expand_k:
                break

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
