"""Dataset class for the Indiana University Chest X-ray Collection (IU X-ray /
Open-I). Expects a manifest CSV built by scripts/setup_iu_xray.py with columns:
image_path, patient_id, report_text, mesh_tags (semicolon-separated)."""

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class IUXrayDataset(Dataset):
    def __init__(self, manifest_csv: str = None, image_root: str = None, transform=None,
                 mode: str = "pairs", label_vocab: list = None, df: pd.DataFrame = None):
        """mode='pairs': returns (image, report_text, patient_id) for Stage 1
        contrastive pretraining, no labels required.
        mode='labeled': also returns a MeSH-derived multi-hot label vector —
        requires label_vocab (see label_schema.build_mesh_vocab).
        Pass either manifest_csv (loads from disk) or a pre-split df directly."""
        if mode not in ("pairs", "labeled"):
            raise ValueError(f"mode must be 'pairs' or 'labeled', got {mode!r}")
        if mode == "labeled" and label_vocab is None:
            raise ValueError("label_vocab is required when mode='labeled'")
        if df is None and manifest_csv is None:
            raise ValueError("must pass either manifest_csv or df")

        self.df = df.reset_index(drop=True) if df is not None else pd.read_csv(manifest_csv)
        self.image_root = image_root
        self.transform = transform
        self.mode = mode
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.df)

    def _mesh_tags(self, row) -> list:
        raw = row["mesh_tags"]
        if pd.isna(raw) or raw == "":
            return []
        return str(raw).split(";")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{self.image_root}/{row['image_path']}").convert("RGB")
        if self.transform:
            image = self.transform(image)
        report_text = "" if pd.isna(row["report_text"]) else str(row["report_text"])

        if self.mode == "pairs":
            return image, report_text, row["patient_id"]

        from .label_schema import mesh_tags_to_vector
        label_vec = mesh_tags_to_vector(self._mesh_tags(row), self.label_vocab)
        return image, report_text, label_vec, row["patient_id"]
