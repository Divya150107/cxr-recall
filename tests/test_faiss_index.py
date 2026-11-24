import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.faiss_index import build_index, same_study_rate, search, search_excluding_study


def _two_view_pool(num_studies=6, dim=8, seed=0):
    """num_studies studies x 2 near-identical views each (a frontal/lateral
    pair whose embeddings are close), so the plain nearest neighbor of any
    row is its own study-mate -- exactly the leakage case."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(num_studies, dim))
    pool = np.concatenate([centers + 0.01 * rng.normal(size=(num_studies, dim)),
                           centers + 0.01 * rng.normal(size=(num_studies, dim))])
    study_ids = np.array([f"s{i}" for i in range(num_studies)] * 2)
    return pool.astype(np.float32), study_ids


def test_plain_search_returns_study_mate_first_when_query_is_in_pool():
    # documents the leakage that search_excluding_study exists to remove
    pool, study_ids = _two_view_pool()
    index = build_index(pool)
    _scores, idx = search(index, pool, 3)
    # rank 0 is the query row itself, rank 1 its study-mate
    assert (idx[:, 0] == np.arange(len(pool))).all()
    assert (study_ids[idx[:, 1]] == study_ids).all()


def test_excluding_study_drops_self_and_study_mate():
    pool, study_ids = _two_view_pool()
    index = build_index(pool)
    k = 3
    scores, idx = search_excluding_study(index, pool, study_ids, study_ids, k)
    assert idx.shape == (len(pool), k)
    assert scores.shape == (len(pool), k)
    for i in range(len(pool)):
        assert i not in idx[i]
        assert (study_ids[idx[i]] != study_ids[i]).all()
    assert same_study_rate(idx, study_ids, study_ids) == 0.0


def test_excluding_study_keeps_similarity_order():
    pool, study_ids = _two_view_pool()
    index = build_index(pool)
    scores, idx = search_excluding_study(index, pool, study_ids, study_ids, 3)
    assert (np.diff(scores, axis=1) <= 1e-6).all()  # descending cosine similarity
    # the survivors are exactly the plain top-k after removing same-study rows
    plain_scores, plain_idx = search(index, pool, len(pool))
    for i in range(len(pool)):
        expected = [j for j in plain_idx[i] if study_ids[j] != study_ids[i]][:3]
        assert list(idx[i]) == expected


def test_no_exclusion_when_queries_share_no_study_with_pool():
    # val-vs-train usage: identical to plain search
    pool, study_ids = _two_view_pool()
    index = build_index(pool)
    rng = np.random.default_rng(1)
    queries = rng.normal(size=(4, pool.shape[1])).astype(np.float32)
    query_ids = np.array(["v0", "v1", "v2", "v3"])
    plain_scores, plain_idx = search(index, queries, 3)
    scores, idx = search_excluding_study(index, queries, query_ids, study_ids, 3)
    assert (idx == plain_idx).all()
    assert np.allclose(scores, plain_scores)


def test_raises_when_not_enough_other_study_rows():
    pool, study_ids = _two_view_pool(num_studies=2)  # 4 rows, only 2 outside any given study
    index = build_index(pool)
    with pytest.raises(ValueError):
        search_excluding_study(index, pool, study_ids, study_ids, 3)


def test_same_study_rate_counts_queries_with_any_hit():
    study_ids = np.array(["a", "a", "b", "b"])
    neighbors = np.array([[1, 2], [2, 3], [0, 1], [0, 1]])  # q0 hits a-mate, q1 clean, q2/q3 clean
    assert same_study_rate(neighbors, study_ids, study_ids) == pytest.approx(0.25)
