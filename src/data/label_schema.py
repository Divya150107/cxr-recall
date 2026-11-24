"""Label handling for two regimes: the full unified schema (MIMIC-CXR/CheXpert/
ChestX-ray14, used once those datasets are available) and IU X-ray's MeSH-tag
vocabulary (used now, for the bootstrap classifier)."""

from collections import Counter

import numpy as np

# --- Full unified schema (dormant until MIMIC-CXR/CheXpert/ChestX-ray14 are wired in) ---

SHARED_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

CHESTXRAY14_TO_SHARED = {label: label for label in SHARED_LABELS}

CHEXPERT_TO_SHARED = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Pleural Effusion": "Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    # No CheXpert equivalent: Emphysema, Fibrosis, Hernia, Infiltration, Mass,
    # Nodule, Pleural_Thickening -> left unmapped (treated as missing, not negative)
    # CheXpert-only labels with no shared-schema slot: No Finding, Enlarged
    # Cardiomediastinum, Lung Opacity, Lung Lesion, Pleural Other, Fracture,
    # Support Devices -> dropped during unification
}


def map_chexpert_row(row: dict, uncertain_policy: str = "zeros") -> np.ndarray:
    """CheXpert labels are -1 (uncertain) / 0 (negative) / 1 (positive) / blank
    (not mentioned). uncertain_policy='zeros' maps -1->0, 'ones' maps -1->1."""
    vec = np.full(len(SHARED_LABELS), np.nan, dtype=np.float32)
    for chexpert_label, shared_label in CHEXPERT_TO_SHARED.items():
        idx = SHARED_LABELS.index(shared_label)
        val = row.get(chexpert_label)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        val = float(val)
        if val == -1:
            val = 0.0 if uncertain_policy == "zeros" else 1.0
        vec[idx] = val
    return vec


def map_chestxray14_row(row: dict) -> np.ndarray:
    vec = np.zeros(len(SHARED_LABELS), dtype=np.float32)
    for i, label in enumerate(SHARED_LABELS):
        vec[i] = float(row.get(label, 0))
    return vec


# --- IU X-ray: MeSH-tag vocabulary (used now) ---

def build_mesh_vocab(reports: list, top_k: int = 14) -> list:
    """reports: list of MeSH-tag lists (one per report). Returns the top_k most
    frequent tags, most-frequent first, so results are stable given the same data."""
    counts = Counter(tag for tags in reports for tag in tags)
    return [tag for tag, _ in counts.most_common(top_k)]


def mesh_tags_to_vector(tags: list, vocab: list) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    tag_set = set(tags)
    for i, vocab_tag in enumerate(vocab):
        if vocab_tag in tag_set:
            vec[i] = 1.0
    return vec
