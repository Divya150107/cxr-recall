"""Patient-level train/val/test splitting, dataset-agnostic."""

import hashlib

import numpy as np
import pandas as pd


def patient_level_split(df: pd.DataFrame, patient_id_col: str,
                         ratios: tuple = (0.8, 0.1, 0.1), seed: int = 42):
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    patients = sorted(df[patient_id_col].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    train_df = df[df[patient_id_col].isin(train_patients)]
    val_df = df[df[patient_id_col].isin(val_patients)]
    test_df = df[df[patient_id_col].isin(test_patients)]
    return train_df, val_df, test_df


def check_no_leakage(patient_id_col: str, *dfs: pd.DataFrame) -> None:
    seen = set()
    for df in dfs:
        ids = set(df[patient_id_col].unique())
        overlap = ids & seen
        if overlap:
            raise AssertionError(f"Patient ID leakage across splits: {overlap}")
        seen |= ids


def deterministic_patient_hash(raw_id: str) -> str:
    """Stable, non-reversible patient ID for cases where the raw ID shouldn't
    be persisted verbatim alongside derived artifacts (embeddings, indices)."""
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
