import numpy as np
import pytest

from src.retrieval.feature_weighting import (dimension_weights, plain_cosine, rerank_shortlist, top_k,
                                             weighted_cosine)

rng = np.random.default_rng(0)
Q = rng.normal(size=(6, 16)).astype(np.float32)
X = rng.normal(size=(40, 16)).astype(np.float32)
W = rng.normal(size=(5, 16)).astype(np.float32)
b = rng.normal(size=5).astype(np.float32)


def test_beta_zero_is_plain_cosine():
    w = dimension_weights(Q, W, b, beta=0.0)
    assert np.allclose(weighted_cosine(Q, X, w), plain_cosine(Q, X), atol=1e-6)


def test_top_frac_one_is_plain_cosine():
    w = dimension_weights(Q, W, b, top_frac=1.0)
    assert np.allclose(weighted_cosine(Q, X, w), plain_cosine(Q, X), atol=1e-6)


def test_hard_selection_keeps_requested_fraction():
    w = dimension_weights(Q, W, b, top_frac=0.25)
    assert set(np.unique(w)) <= {0.0, 1.0}
    assert (w.sum(axis=1) == 4).all()


def test_hard_selection_equals_cosine_on_selected_dims():
    w = dimension_weights(Q, W, b, top_frac=0.5)
    sims = weighted_cosine(Q, X, w)
    dims = w[0] == 1
    expected = plain_cosine(Q[:1, dims], X[:, dims])[0]
    assert np.allclose(sims[0], expected, atol=1e-6)


def test_soft_weights_mean_one_at_beta_one():
    w = dimension_weights(Q, W, b, beta=1.0)
    assert np.allclose(w.mean(axis=1), 1.0, atol=1e-5)


def test_weighted_cosine_self_similarity_is_one():
    w = dimension_weights(Q, W, b, beta=1.0)
    assert np.allclose(np.diag(weighted_cosine(Q, Q, w)), 1.0, atol=1e-6)


def test_exactly_one_mode_required():
    with pytest.raises(ValueError):
        dimension_weights(Q, W, b)
    with pytest.raises(ValueError):
        dimension_weights(Q, W, b, beta=1.0, top_frac=0.5)


def test_rerank_only_reorders_the_shortlist():
    plain = plain_cosine(Q, X)
    weighted = weighted_cosine(Q, X, dimension_weights(Q, W, b, beta=1.0))
    out = rerank_shortlist(plain, weighted, shortlist=10, k=5)
    shortlist = top_k(plain, 10)
    for row in range(len(Q)):
        assert set(out[row]) <= set(shortlist[row])
        assert list(out[row]) == sorted(out[row], key=lambda j: -weighted[row, j])
