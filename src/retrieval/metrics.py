"""Retrieval-quality metrics: how clinically relevant are the cases Arm 2/3
retrieve, independent of whether retrieval influences the diagnosis."""

import numpy as np


def label_agreement_at_k(query_labels: np.ndarray, retrieved_labels: np.ndarray) -> float:
    """query_labels: (num_labels,) multi-hot. retrieved_labels: (k, num_labels)
    multi-hot for the k retrieved cases. Fraction of the k sharing at least
    one positive label with the query."""
    if retrieved_labels.shape[0] == 0:
        return float("nan")
    shares_label = (retrieved_labels * query_labels).sum(axis=1) > 0
    return float(shares_label.mean())


def mean_reciprocal_rank(query_labels: np.ndarray, retrieved_labels: np.ndarray) -> float:
    """Reciprocal rank of the first retrieved case sharing a label with the
    query; 0.0 if none of the k do."""
    shares_label = (retrieved_labels * query_labels).sum(axis=1) > 0
    hits = np.nonzero(shares_label)[0]
    if len(hits) == 0:
        return 0.0
    return 1.0 / (hits[0] + 1)


def label_agreement_by_similarity_bucket(top1_similarity: np.ndarray, query_labels: np.ndarray,
                                          retrieved_labels: np.ndarray, num_buckets: int = 5) -> list:
    """Buckets queries into num_buckets equal-*frequency* bins by
    top1_similarity (quantiles via sorted order, not equal-width similarity
    ranges -- cosine similarity on a fine-tuned embedding space tends to
    cluster, so equal-width bins would leave some buckets nearly empty).
    Within each bucket, computes mean label_agreement_at_k across its
    queries. Returns a list of dicts ordered low-to-high similarity:
    {similarity_range, mean_top1_sim, mean_label_agreement, count}. Used to
    sanity-check a calibrated confidence threshold against this project's
    random-retrieval baseline: below the threshold, retrieved evidence
    should look statistically no more label-relevant than chance."""
    order = np.argsort(top1_similarity)
    buckets = np.array_split(order, num_buckets)
    results = []
    for bucket_indices in buckets:
        agreements = [label_agreement_at_k(query_labels[i], retrieved_labels[i]) for i in bucket_indices]
        sims = top1_similarity[bucket_indices]
        results.append({
            "similarity_range": (float(sims.min()), float(sims.max())),
            "mean_top1_sim": float(sims.mean()),
            "mean_label_agreement": float(np.nanmean(agreements)),
            "count": len(bucket_indices),
        })
    return results
