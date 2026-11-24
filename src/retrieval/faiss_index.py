"""FAISS-based retrieval index. Arm 2 uses this for post-hoc, display-only
retrieval; Arm 3 (CXR-Recall) reuses the same index as the first stage of
its pre-diagnosis evidence retrieval."""

import numpy as np
import faiss


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """L2-normalizes embeddings and builds a flat inner-product (cosine
    similarity) index -- exact search, appropriate at this dataset's scale
    (thousands, not millions, of vectors)."""
    embeddings = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(index: faiss.Index, queries: np.ndarray, k: int):
    queries = queries.astype(np.float32).copy()
    faiss.normalize_L2(queries)
    scores, indices = index.search(queries, k)
    return scores, indices


def search_excluding_study(index: faiss.Index, queries: np.ndarray, query_study_ids: np.ndarray,
                           pool_study_ids: np.ndarray, k: int):
    """search(), but drops any neighbor from the same study as the query and
    returns exactly k of the rest. Needed for train-vs-train retrieval (the
    evidence Arm 3 trains on): IU X-ray studies mostly have 2+ views (frontal
    + lateral) sharing one report and one label vector, so excluding only the
    query's own row (index-only self-exclusion) leaves its study-mate in the
    pool -- a neighbor whose label token IS the query's label. Val/test
    queries can never retrieve a study-mate (the split is study-level), so
    index-only exclusion trained the evidence encoder under a regime it
    never sees at eval time.

    pool_study_ids: (n_pool,) study id per index row, in index order.
    query_study_ids: (n_query,) study id per query. Works unchanged for
    val-vs-train queries (nothing shares a study, so nothing is dropped).
    Over-fetches k + max_views_per_study so k survivors are guaranteed
    whenever the pool has at least that many other-study rows; raises
    ValueError otherwise rather than silently returning fewer neighbors."""
    query_study_ids = np.asarray(query_study_ids)
    pool_study_ids = np.asarray(pool_study_ids)
    _ids, counts = np.unique(pool_study_ids, return_counts=True)
    fetch_k = k + int(counts.max())
    scores, indices = search(index, queries, fetch_k)

    out_scores = np.empty((len(queries), k), dtype=scores.dtype)
    out_indices = np.empty((len(queries), k), dtype=indices.dtype)
    for i, (row, row_scores) in enumerate(zip(indices, scores)):
        keep = [(j, s) for j, s in zip(row, row_scores) if pool_study_ids[j] != query_study_ids[i]][:k]
        if len(keep) < k:
            raise ValueError(f"query {i}: only {len(keep)} neighbors outside its study in the top {fetch_k}; "
                             f"need {k}")
        out_indices[i] = [j for j, _ in keep]
        out_scores[i] = [s for _, s in keep]
    return out_scores, out_indices


def same_study_rate(neighbor_indices: np.ndarray, query_study_ids: np.ndarray,
                    pool_study_ids: np.ndarray) -> float:
    """Diagnostic: fraction of queries with at least one retrieved neighbor
    from the query's own study. Run it on the index-only-excluded neighbor
    set to measure how much same-study leakage a prior run actually had."""
    query_study_ids = np.asarray(query_study_ids)
    pool_study_ids = np.asarray(pool_study_ids)
    hits = (pool_study_ids[neighbor_indices] == query_study_ids[:, None]).any(axis=1)
    return float(hits.mean()) if len(hits) else float("nan")
