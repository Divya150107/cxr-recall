import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.metrics import (label_agreement_at_k, label_agreement_by_similarity_bucket,
                                    mean_reciprocal_rank)


def test_label_agreement_at_k_all_match():
    query = np.array([1.0, 0.0, 0.0])
    retrieved = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    assert label_agreement_at_k(query, retrieved) == 2 / 3


def test_label_agreement_at_k_no_match():
    query = np.array([1.0, 0.0, 0.0])
    retrieved = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert label_agreement_at_k(query, retrieved) == 0.0


def test_label_agreement_at_k_empty_retrieval():
    query = np.array([1.0, 0.0, 0.0])
    retrieved = np.zeros((0, 3))
    assert np.isnan(label_agreement_at_k(query, retrieved))


def test_mean_reciprocal_rank_first_hit():
    query = np.array([1.0, 0.0])
    retrieved = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert mean_reciprocal_rank(query, retrieved) == 1.0


def test_mean_reciprocal_rank_third_hit():
    query = np.array([0.0, 1.0])
    retrieved = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert mean_reciprocal_rank(query, retrieved) == 1 / 3


def test_mean_reciprocal_rank_no_hit():
    query = np.array([1.0, 0.0])
    retrieved = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert mean_reciprocal_rank(query, retrieved) == 0.0


# --- similarity-bucket calibration ---

def test_buckets_partition_all_queries_exactly_once():
    n, num_buckets = 20, 5
    rng = np.random.default_rng(0)
    top1_similarity = rng.uniform(0, 1, size=n)
    query_labels = np.tile([1.0, 0.0], (n, 1))
    retrieved_labels = np.tile([1.0, 0.0], (n, 3, 1))

    buckets = label_agreement_by_similarity_bucket(top1_similarity, query_labels, retrieved_labels, num_buckets)
    assert sum(b["count"] for b in buckets) == n
    assert len(buckets) == num_buckets


def test_buckets_ordered_low_to_high_similarity():
    n = 20
    top1_similarity = np.linspace(0, 1, n)
    query_labels = np.tile([1.0, 0.0], (n, 1))
    retrieved_labels = np.tile([1.0, 0.0], (n, 3, 1))

    buckets = label_agreement_by_similarity_bucket(top1_similarity, query_labels, retrieved_labels, num_buckets=5)
    sims = [b["mean_top1_sim"] for b in buckets]
    assert sims == sorted(sims)


def test_buckets_capture_similarity_agreement_trend():
    # low-similarity queries constructed to never share a label with their
    # retrieved neighbors, high-similarity queries constructed to always
    # share one -- the bucket trend must reflect that, not just partition data
    n = 10
    top1_similarity = np.linspace(0, 1, n)
    query_labels = np.tile([1.0, 0.0], (n, 1))
    retrieved_labels = np.array([[[0.0, 1.0]] * 3 if i < n // 2 else [[1.0, 0.0]] * 3 for i in range(n)])

    buckets = label_agreement_by_similarity_bucket(top1_similarity, query_labels, retrieved_labels, num_buckets=2)
    assert buckets[0]["mean_label_agreement"] == 0.0   # low-similarity bucket
    assert buckets[1]["mean_label_agreement"] == 1.0   # high-similarity bucket
