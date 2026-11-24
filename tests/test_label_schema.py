import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.label_schema import (
    CHEXPERT_TO_SHARED,
    SHARED_LABELS,
    build_mesh_vocab,
    map_chestxray14_row,
    map_chexpert_row,
    mesh_tags_to_vector,
)


def test_build_mesh_vocab_is_deterministic_and_frequency_ordered():
    reports = [
        ["Cardiomegaly", "Opacity"],
        ["Cardiomegaly"],
        ["Opacity", "Effusion"],
        ["Effusion"],
        ["Effusion"],
    ]
    vocab = build_mesh_vocab(reports, top_k=2)
    assert vocab == ["Effusion", "Cardiomegaly"] or vocab == ["Effusion", "Opacity"]
    assert vocab[0] == "Effusion"  # most frequent (3 occurrences) must be first


def test_mesh_tags_to_vector_shape_and_values():
    vocab = ["Effusion", "Cardiomegaly", "Opacity"]
    vec = mesh_tags_to_vector(["Cardiomegaly", "Nodule"], vocab)
    assert vec.shape == (3,)
    assert list(vec) == [0.0, 1.0, 0.0]


def test_map_chexpert_row_uncertain_policy():
    row = {"Atelectasis": 1, "Cardiomegaly": -1, "Pleural Effusion": 0}
    vec_zeros = map_chexpert_row(row, uncertain_policy="zeros")
    vec_ones = map_chexpert_row(row, uncertain_policy="ones")

    atel_idx = SHARED_LABELS.index("Atelectasis")
    cardio_idx = SHARED_LABELS.index("Cardiomegaly")
    effusion_idx = SHARED_LABELS.index("Effusion")

    assert vec_zeros[atel_idx] == 1.0
    assert vec_zeros[cardio_idx] == 0.0
    assert vec_ones[cardio_idx] == 1.0
    assert vec_zeros[effusion_idx] == 0.0

    # a label never mentioned in the row must stay NaN (missing), not 0
    hernia_idx = SHARED_LABELS.index("Hernia")
    assert np.isnan(vec_zeros[hernia_idx])


def test_map_chestxray14_row_identity_mapping():
    row = {label: 0 for label in SHARED_LABELS}
    row["Pneumonia"] = 1
    vec = map_chestxray14_row(row)
    assert vec[SHARED_LABELS.index("Pneumonia")] == 1.0
    assert vec.sum() == 1.0


def test_chexpert_mapping_only_targets_shared_labels():
    assert set(CHEXPERT_TO_SHARED.values()).issubset(set(SHARED_LABELS))
