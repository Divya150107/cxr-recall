"""Arm-1 (no-retrieval baseline) training on IU X-ray. This is a pipeline
smoke test against coarse MeSH-tag labels, not the thesis's real Arm-1 result
(that needs the full unified schema on MIMIC-CXR/CheXpert/ChestX-ray14 — see
src/data/label_schema.py and the CXR-Recall proposal, Section 6.6/6.7).

Usage:
    python train.py --config configs/arm1_iu_xray.yaml
    python train.py --config configs/arm1_iu_xray.yaml --epochs 1 --limit 50
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.models import ViT_B_16_Weights

from src.data.iu_xray import IUXrayDataset
from src.data.label_schema import build_mesh_vocab, mesh_tags_to_vector
from src.data.splits import check_no_leakage, patient_level_split
from src.models.classifier import NoRetrievalClassifier


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            continue  # AUROC undefined for a label with only one class in this split
        scores.append(roc_auc_score(col, y_score[:, i]))
    return float(np.mean(scores)) if scores else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override config")
    parser.add_argument("--limit", type=int, default=None,
                         help="truncate dataset to N rows, for a fast smoke test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    full_df = pd.read_csv(cfg["data"]["manifest_csv"])
    if args.limit:
        full_df = full_df.head(args.limit)

    train_df, val_df, test_df = patient_level_split(
        full_df, cfg["data"]["patient_id_col"],
        tuple(cfg["data"]["split_ratios"]), cfg["data"]["split_seed"])
    check_no_leakage(cfg["data"]["patient_id_col"], train_df, val_df, test_df)
    print(f"split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_mesh_lists = [str(t).split(";") if pd.notna(t) else [] for t in train_df["mesh_tags"]]
    vocab = build_mesh_vocab(train_mesh_lists, top_k=cfg["data"]["label_vocab_size"])
    print(f"label vocab ({len(vocab)} tags): {vocab}")

    transform = ViT_B_16_Weights.DEFAULT.transforms()
    image_root = cfg["data"]["image_root"]

    train_ds = IUXrayDataset(df=train_df, image_root=image_root, transform=transform,
                              mode="labeled", label_vocab=vocab)
    val_ds = IUXrayDataset(df=val_df, image_root=image_root, transform=transform,
                            mode="labeled", label_vocab=vocab)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                               shuffle=True, num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                             shuffle=False, num_workers=cfg["train"]["num_workers"])

    model = NoRetrievalClassifier(num_labels=len(vocab),
                                   pretrained=cfg["model"]["pretrained"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    epochs = args.epochs or cfg["train"]["epochs"]
    checkpoint_dir = cfg["train"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, _texts, labels, _pids in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= max(len(train_ds), 1)

        model.eval()
        all_labels, all_scores = [], []
        with torch.no_grad():
            for images, _texts, labels, _pids in val_loader:
                images = images.to(device)
                probs = torch.sigmoid(model(images)).cpu().numpy()
                all_labels.append(labels.numpy())
                all_scores.append(probs)

        if all_labels:
            y_true = np.concatenate(all_labels)
            y_score = np.concatenate(all_scores)
            val_auroc = macro_auroc(y_true, y_score)
        else:
            val_auroc = float("nan")

        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_macro_auroc={val_auroc:.4f}")

    ckpt_path = os.path.join(checkpoint_dir, "arm1_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "label_vocab": vocab}, ckpt_path)
    print(f"saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
