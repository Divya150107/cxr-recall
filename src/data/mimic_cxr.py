"""Stub for MIMIC-CXR(-JPG), matching IUXrayDataset's interface. Activate once
PhysioNet credentialing (see the CXR-Recall proposal, Section 8.1) is approved:
point manifest_csv/image_root at the real MIMIC-CXR-JPG layout and remove the
NotImplementedError below. No other code needs to change — train.py and the
model classes consume this the same way they consume IUXrayDataset."""

from torch.utils.data import Dataset


class MimicCxrDataset(Dataset):
    def __init__(self, manifest_csv: str, image_root: str, transform=None,
                 mode: str = "pairs", label_vocab: list = None):
        raise NotImplementedError(
            "MIMIC-CXR access is pending PhysioNet credentialing. Once approved: "
            "build a manifest CSV (image_path, patient_id, report_text, and either "
            "mesh_tags or the CheXpert-style 14-label columns), then implement this "
            "the same way as IUXrayDataset in src/data/iu_xray.py, using "
            "label_schema.map_chexpert_row for the 'labeled' mode instead of "
            "mesh_tags_to_vector."
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
