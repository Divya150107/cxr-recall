"""Arm 2: post-hoc retrieval index + evaluation. Builds a FAISS index over
the Arm-1-trained encoder's embeddings for the training set, then measures
retrieval quality against the validation set. This does NOT feed retrieved
evidence into a diagnosis -- it only tests whether the retrieved cases are
clinically relevant, which is what post-hoc retrieval (RadFuse-style)
actually offers. Arm 3 reuses this same index as its first retrieval pass.

Usage:
    python retrieve.py --config configs/arm1_iu_xray.yaml \
        --checkpoint checkpoints/arm1_iu_xray/arm1_final.pt --k 5
"""

import argparse
import os

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models import ViT_B_16_Weights

from src.data.iu_xray import IUXrayDataset
from src.data.splits import patient_level_split
from src.models.encoders import ImageEncoder
from src.retrieval.embeddings import extract_embeddings
from src.retrieval.faiss_index import build_index, search
from src.retrieval.metrics import label_agreement_at_k, mean_reciprocal_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    label_vocab = ckpt["label_vocab"]

    encoder = ImageEncoder(pretrained=False).to(device)
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["model_state_dict"].items()
                      if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)

    full_df = pd.read_csv(cfg["data"]["manifest_csv"])
    train_df, val_df, _test_df = patient_level_split(
        full_df, cfg["data"]["patient_id_col"],
        tuple(cfg["data"]["split_ratios"]), cfg["data"]["split_seed"])

    transform = ViT_B_16_Weights.DEFAULT.transforms()
    image_root = cfg["data"]["image_root"]
    train_ds = IUXrayDataset(df=train_df, image_root=image_root, transform=transform,
                              mode="labeled", label_vocab=label_vocab)
    val_ds = IUXrayDataset(df=val_df, image_root=image_root, transform=transform,
                            mode="labeled", label_vocab=label_vocab)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                               shuffle=False, num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                             shuffle=False, num_workers=cfg["train"]["num_workers"])

    print("extracting train embeddings...")
    train_embeddings, train_labels = extract_embeddings(encoder, train_loader, device)
    print("extracting val embeddings...")
    val_embeddings, val_labels = extract_embeddings(encoder, val_loader, device)

    index = build_index(train_embeddings)
    _scores, neighbor_idx = search(index, val_embeddings, args.k)

    agreements, mrrs = [], []
    for i in range(len(val_embeddings)):
        retrieved_labels = train_labels[neighbor_idx[i]]
        agreements.append(label_agreement_at_k(val_labels[i], retrieved_labels))
        mrrs.append(mean_reciprocal_rank(val_labels[i], retrieved_labels))

    print(f"k={args.k}  label_agreement@k={np.mean(agreements):.4f}  MRR={np.mean(mrrs):.4f}")

    out_dir = os.path.dirname(args.checkpoint)
    faiss.write_index(index, os.path.join(out_dir, "arm2_retrieval.index"))
    np.save(os.path.join(out_dir, "arm2_train_labels.npy"), train_labels)
    print(f"saved retrieval index -> {out_dir}/arm2_retrieval.index")


if __name__ == "__main__":
    main()
