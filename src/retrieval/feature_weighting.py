"""Query-adaptive "selective feature" retrieval (Idea 2).

Plain retrieval compares the whole image embedding with cosine similarity,
so every dimension counts equally. Most of those dimensions have nothing to
do with the query's likely finding (they encode other findings, view,
devices, scanner style), and in this project the embedding space is very
compressed (median top-1 cosine ~0.99 in Phase 1), so whole-embedding
similarity barely separates cases. Idea 2: compare only the features that
matter for this query.

Which features matter comes from a linear probe W (num_labels x dim) fit on
TRAIN embeddings only. For a query with predicted label probabilities
p = sigmoid(W q + b), dimension d gets relevance

    r_d = sum_c p_c |W_cd| / sum_c p_c        (rescaled to mean 1)

and similarity becomes a weighted cosine over the weights w:

    cos_w(q, x) = sum_d w_d^2 q_d x_d / (||w*q|| ||w*x||)

Two ways to turn r into w:
  - soft  (beta):     w = (1 - beta) + beta * r.  beta=0 is plain cosine.
  - hard  (top_frac): w = 1 on the top `top_frac` fraction of dimensions by r,
                      0 elsewhere -- literally comparing only selected
                      features. top_frac=1 is plain cosine.

And "selected images": rerank_shortlist() keeps the plain-cosine top-N
shortlist and only reorders it by the weighted similarity.

Caveat for Arm 3: the probe is fit on train rows, so p for a TRAIN query is
in-sample. That is fine for val/test queries (all evaluation here), but
building Arm 3's train-time evidence with this would need cross-fitting.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_probe(train_emb: np.ndarray, train_labels: np.ndarray, C: float = 1.0, max_iter: int = 2000):
    """One logistic regression per label on TRAIN embeddings. Returns
    (W, b) as float32 arrays of shape (num_labels, dim), (num_labels,).
    Labels with a single class in train get a zero row (no signal)."""
    num_labels, dim = train_labels.shape[1], train_emb.shape[1]
    W = np.zeros((num_labels, dim), dtype=np.float32)
    b = np.zeros(num_labels, dtype=np.float32)
    for c in range(num_labels):
        y = train_labels[:, c]
        if y.min() == y.max():
            continue
        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(train_emb, y)
        W[c], b[c] = clf.coef_[0], clf.intercept_[0]
    return W, b


def predict_proba(emb: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(emb @ W.T + b)))


def dimension_weights(query_emb: np.ndarray, W: np.ndarray, b: np.ndarray,
                      beta: float = None, top_frac: float = None) -> np.ndarray:
    """Per-query dimension weights, shape (n_queries, dim). Pass exactly one
    of beta (soft) or top_frac (hard)."""
    if (beta is None) == (top_frac is None):
        raise ValueError("pass exactly one of beta or top_frac")
    p = predict_proba(query_emb, W, b)                       # (n, L)
    r = (p @ np.abs(W)) / p.sum(axis=1, keepdims=True)       # (n, D)
    r = r / r.mean(axis=1, keepdims=True)
    if beta is not None:
        return ((1.0 - beta) + beta * r).astype(np.float32)
    n_keep = max(1, int(round(top_frac * r.shape[1])))
    keep = np.argsort(-r, axis=1)[:, :n_keep]
    w = np.zeros_like(r, dtype=np.float32)
    np.put_along_axis(w, keep, 1.0, axis=1)
    return w


def weighted_cosine(query_emb: np.ndarray, corpus_emb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(n_queries, n_corpus) weighted cosine, each query with its own weights."""
    w2 = weights.astype(np.float64) ** 2
    q = query_emb.astype(np.float64)
    X = corpus_emb.astype(np.float64)
    num = (q * w2) @ X.T
    corpus_norm = np.sqrt(w2 @ (X ** 2).T)                     # (n, N)
    query_norm = np.sqrt((w2 * q ** 2).sum(axis=1, keepdims=True))
    denom = corpus_norm * query_norm
    denom[denom == 0] = 1.0
    return num / denom


def plain_cosine(query_emb: np.ndarray, corpus_emb: np.ndarray) -> np.ndarray:
    q = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    X = corpus_emb / np.linalg.norm(corpus_emb, axis=1, keepdims=True)
    return q @ X.T


def top_k(sims: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first."""
    return np.argsort(-sims, axis=1, kind="stable")[:, :k]


def rerank_shortlist(plain_sims: np.ndarray, weighted_sims: np.ndarray, shortlist: int, k: int) -> np.ndarray:
    """Two-stage: shortlist the top `shortlist` by plain cosine, then return
    the top k of that shortlist ordered by weighted similarity."""
    cand = top_k(plain_sims, shortlist)                                    # (n, S)
    w_on_cand = np.take_along_axis(weighted_sims, cand, axis=1)
    order = np.argsort(-w_on_cand, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(cand, order, axis=1)
