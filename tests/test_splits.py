import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import check_no_leakage, patient_level_split


def make_df(n_patients=100, rows_per_patient=2):
    rows = []
    for pid in range(n_patients):
        for _ in range(rows_per_patient):
            rows.append({"patient_id": f"p{pid}", "value": pid})
    return pd.DataFrame(rows)


def test_split_ratios_sum_validation():
    df = make_df()
    with pytest.raises(ValueError):
        patient_level_split(df, "patient_id", ratios=(0.8, 0.1, 0.2))


def test_no_leakage_across_splits():
    df = make_df(n_patients=100, rows_per_patient=3)
    train_df, val_df, test_df = patient_level_split(df, "patient_id", seed=42)
    check_no_leakage("patient_id", train_df, val_df, test_df)  # must not raise

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)


def test_split_is_deterministic_given_same_seed():
    df = make_df()
    train1, val1, test1 = patient_level_split(df, "patient_id", seed=7)
    train2, val2, test2 = patient_level_split(df, "patient_id", seed=7)
    assert set(train1["patient_id"]) == set(train2["patient_id"])
    assert set(val1["patient_id"]) == set(val2["patient_id"])
    assert set(test1["patient_id"]) == set(test2["patient_id"])


def test_check_no_leakage_detects_injected_overlap():
    df = make_df(n_patients=10, rows_per_patient=1)
    train_df, val_df, test_df = patient_level_split(df, "patient_id", seed=1)
    corrupted_val = pd.concat([val_df, train_df.head(1)])
    with pytest.raises(AssertionError):
        check_no_leakage("patient_id", train_df, corrupted_val, test_df)


def test_split_respects_approximate_ratios():
    df = make_df(n_patients=1000, rows_per_patient=1)
    train_df, val_df, test_df = patient_level_split(df, "patient_id", ratios=(0.8, 0.1, 0.1), seed=42)
    assert abs(len(train_df) - 800) <= 5
    assert abs(len(val_df) - 100) <= 5
    assert abs(len(test_df) - 100) <= 5
