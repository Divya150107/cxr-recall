"""Evaluate Idea 2 (selective-feature retrieval, src/retrieval/feature_weighting.py)
on cached image embeddings, CPU only.

Input: a directory with {train,val,test}_img.npy and {train,val,test}_lbl.npy,
e.g. the unzipped phase2_stage1_embeddings.zip from the Phase 2 notebook.

Protocol (no setting is chosen on test):
  1. Fit the linear probe on TRAIN, choosing C by val macro AUROC.
  2. Sweep variants on VAL queries retrieving from the TRAIN index:
     plain cosine; soft weighting (beta); hard selection (top_frac);
     shortlist re-rank. Metric = Arm 2's label_agreement@5, plus MRR.
  3. Take the single best variant by val label_agreement@5 and compare it
     with plain cosine once on TEST: paired Wilcoxon + paired bootstrap 95% CI.

Val/test queries cannot retrieve their own study (the split is
study-level), so no study exclusion is needed here.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.retrieval.feature_weighting import (dimension_weights, fit_probe, plain_cosine, predict_proba,
                                             rerank_shortlist, top_k, weighted_cosine)
from src.retrieval.metrics import label_agreement_at_k, mean_reciprocal_rank

K = 5
SEED = 42
SOFT_BETAS = (0.5, 1.0, 2.0)
HARD_FRACS = (0.1, 0.25, 0.5)
SHORTLISTS = (20, 50)
PROBE_CS = (0.1, 1.0, 10.0, 100.0)


def macro_auroc(labels, scores):
    cols = [c for c in range(labels.shape[1]) if 0 < labels[:, c].sum() < len(labels)]
    return float(np.mean([roc_auc_score(labels[:, c], scores[:, c]) for c in cols]))


def per_query_scores(neighbors, q_lbl, train_lbl):
    agree = np.array([label_agreement_at_k(q_lbl[i], train_lbl[neighbors[i]]) for i in range(len(q_lbl))])
    mrr = np.array([mean_reciprocal_rank(q_lbl[i], train_lbl[neighbors[i]]) for i in range(len(q_lbl))])
    return agree, mrr


def random_baseline(q_lbl, train_lbl):
    return float(np.mean([((train_lbl * q).sum(axis=1) > 0).mean() for q in q_lbl]))


def neighbors_for(variant, q_emb, train_emb, W, b):
    kind, value = variant
    plain = plain_cosine(q_emb, train_emb)
    if kind == "plain":
        return top_k(plain, K)
    if kind == "soft":
        return top_k(weighted_cosine(q_emb, train_emb, dimension_weights(q_emb, W, b, beta=value)), K)
    if kind == "hard":
        return top_k(weighted_cosine(q_emb, train_emb, dimension_weights(q_emb, W, b, top_frac=value)), K)
    if kind == "rerank":  # value = (shortlist, inner variant)
        shortlist, (inner_kind, inner_value) = value
        w = dimension_weights(q_emb, W, b, **({"beta": inner_value} if inner_kind == "soft"
                                               else {"top_frac": inner_value}))
        return rerank_shortlist(plain, weighted_cosine(q_emb, train_emb, w), shortlist, K)
    raise ValueError(kind)


def describe(variant):
    kind, value = variant
    if kind == "plain":
        return "plain cosine"
    if kind == "soft":
        return f"soft beta={value}"
    if kind == "hard":
        return f"hard top {value:.0%} dims"
    shortlist, inner = value
    return f"rerank top-{shortlist} by {describe(inner)}"


def paired_bootstrap(diffs, n=10000, seed=SEED):
    rng = np.random.RandomState(seed)
    means = np.array([diffs[rng.randint(0, len(diffs), len(diffs))].mean() for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi, min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=Path, required=True)
    args = ap.parse_args()
    d = {f"{s}_{t}": np.load(args.emb_dir / f"{s}_{t}.npy") for s in ("train", "val", "test") for t in ("img", "lbl")}
    tr_e, tr_l = d["train_img"], d["train_lbl"]

    # 1. probe, C chosen on val
    best = None
    for C in PROBE_CS:
        W, b = fit_probe(tr_e, tr_l, C=C)
        auc = macro_auroc(d["val_lbl"], predict_proba(d["val_img"], W, b))
        print(f"probe C={C:<6} val macro AUROC={auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, C, W, b)
    _, C, W, b = best
    print(f"-> probe C={C}: val {best[0]:.4f}, test "
          f"{macro_auroc(d['test_lbl'], predict_proba(d['test_img'], W, b)):.4f}")

    # 2. val sweep
    variants = [("plain", None)] + [("soft", x) for x in SOFT_BETAS] + [("hard", x) for x in HARD_FRACS]
    print(f"\nVAL ({len(d['val_lbl'])} queries -> {len(tr_l)} train), random baseline "
          f"{random_baseline(d['val_lbl'], tr_l):.4f}")
    val_scores = {}
    for v in variants:
        agree, mrr = per_query_scores(neighbors_for(v, d["val_img"], tr_e, W, b), d["val_lbl"], tr_l)
        val_scores[v] = agree.mean()
        print(f"  {describe(v):34s} label_agreement@{K}={agree.mean():.4f}  MRR={mrr.mean():.4f}")
    best_inner = max((v for v in variants if v[0] != "plain"), key=lambda v: val_scores[v])
    for s in SHORTLISTS:
        v = ("rerank", (s, best_inner))
        agree, mrr = per_query_scores(neighbors_for(v, d["val_img"], tr_e, W, b), d["val_lbl"], tr_l)
        val_scores[v] = agree.mean()
        print(f"  {describe(v):34s} label_agreement@{K}={agree.mean():.4f}  MRR={mrr.mean():.4f}")

    chosen = max((v for v in val_scores if v[0] != "plain"), key=lambda v: val_scores[v])
    print(f"\nchosen on val: {describe(chosen)} ({val_scores[chosen]:.4f} vs plain "
          f"{val_scores[('plain', None)]:.4f})")

    # 3. one paired test comparison
    print(f"\nTEST ({len(d['test_lbl'])} queries), random baseline {random_baseline(d['test_lbl'], tr_l):.4f}")
    base = per_query_scores(neighbors_for(("plain", None), d["test_img"], tr_e, W, b), d["test_lbl"], tr_l)
    new = per_query_scores(neighbors_for(chosen, d["test_img"], tr_e, W, b), d["test_lbl"], tr_l)
    for name, i in ((f"label_agreement@{K}", 0), ("MRR", 1)):
        diffs = new[i] - base[i]
        lo, hi, p_boot = paired_bootstrap(diffs)
        p_w = stats.wilcoxon(diffs).pvalue if np.any(diffs != 0) else 1.0
        verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant"
        print(f"  {name:20s} plain={base[i].mean():.4f} {describe(chosen)}={new[i].mean():.4f} "
              f"diff={diffs.mean():+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}] Wilcoxon p={p_w:.4g} "
              f"boot p~{p_boot:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
