"""Per-query retrieval-confidence gating: decides whether to trust a query's
retrieved evidence based on its top-1 (best) neighbor's cosine similarity, and
reports which queries fell below threshold for human-review flagging. Kept
separate from metrics.py because this is a runtime trust *decision*, not a
quality *measurement* -- it must stay free of any ground-truth-label
dependency, since real inference has no labels to check retrieval against.
Threshold itself is calibrated empirically -- see
metrics.py:label_agreement_by_similarity_bucket and the notebook's
calibration cell -- not hardcoded here."""

import numpy as np


def confidence_mask(top1_similarity: np.ndarray, threshold: float) -> np.ndarray:
    """top1_similarity: (n,) cosine similarity of each query's best (rank-0)
    retrieved neighbor, e.g. scores[:, 0] from faiss_index.search(). Returns
    (n,) float32 array of 1.0 (trust evidence) / 0.0 (fall back to
    vision-only) -- directly usable as RetrievalFusionClassifier.forward's
    evidence_confidence after torch.from_numpy(...). Threshold is inclusive:
    a query exactly at the threshold is trusted."""
    return (top1_similarity >= threshold).astype(np.float32)


def flag_low_confidence(top1_similarity: np.ndarray, threshold: float) -> dict:
    """Human-review reporting for a batch/val set: which query indices had no
    trustworthy evidence, and what fraction of the set that represents. Does
    NOT itself withhold a prediction -- every flagged index still gets a
    vision-only prediction from the caller; this only drives the review-flag
    surfaced alongside it."""
    below = top1_similarity < threshold
    return {
        "flagged_indices": np.nonzero(below)[0],
        "num_flagged": int(below.sum()),
        "num_total": int(len(top1_similarity)),
        "fraction_flagged": float(below.mean()) if len(top1_similarity) else float("nan"),
    }
