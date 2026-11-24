import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.confidence import confidence_mask, flag_low_confidence


def test_confidence_mask_above_and_below_threshold():
    sims = np.array([0.9, 0.5, 0.2, 0.8])
    mask = confidence_mask(sims, threshold=0.6)
    assert list(mask) == [1.0, 0.0, 0.0, 1.0]


def test_confidence_mask_boundary_is_inclusive():
    sims = np.array([0.6])
    assert confidence_mask(sims, threshold=0.6)[0] == 1.0


def test_confidence_mask_all_pass_or_all_fail():
    sims = np.array([0.9, 0.95])
    assert confidence_mask(sims, threshold=0.1).tolist() == [1.0, 1.0]
    assert confidence_mask(sims, threshold=0.99).tolist() == [0.0, 0.0]


def test_flag_low_confidence_reports_indices_and_fraction():
    sims = np.array([0.9, 0.2, 0.3, 0.8])
    report = flag_low_confidence(sims, threshold=0.5)
    assert list(report["flagged_indices"]) == [1, 2]
    assert report["num_flagged"] == 2
    assert report["num_total"] == 4
    assert report["fraction_flagged"] == 0.5
