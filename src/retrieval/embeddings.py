"""Embedding extraction, shared by Arm 2 (retrieval) and Arm 3 (evidence
encoder input)."""

import numpy as np
import torch


@torch.no_grad()
def extract_embeddings(encoder, dataloader, device) -> tuple:
    """encoder: an ImageEncoder (or any module mapping images -> feature
    vectors). dataloader: yields (images, report_text, labels, patient_ids),
    matching IUXrayDataset's 'labeled' mode."""
    encoder.eval()
    all_embeddings, all_labels = [], []
    for images, _texts, labels, _pids in dataloader:
        images = images.to(device)
        embeddings = encoder(images).cpu().numpy()
        all_embeddings.append(embeddings)
        all_labels.append(labels.numpy())
    return np.concatenate(all_embeddings), np.concatenate(all_labels)


@torch.no_grad()
def extract_dual_embeddings(model, dataloader, device) -> tuple:
    """model: exposes encode_image(images) -> embeddings and
    encode_text(texts, device) -> embeddings (a DualTowerContrastiveModel,
    see src/models/contrastive.py). dataloader: yields (images, report_text,
    labels, patient_ids), matching IUXrayDataset's 'labeled' mode. Computes
    BOTH modalities' frozen embeddings in a single pass -- avoids two full
    passes over the same fixed images/reports. Kept separate from
    extract_embeddings rather than extending it: that function's contract is
    single-encoder, single-tensor-in/out (encoder(images) -> embeddings),
    which a two-modality extraction can't satisfy without changing that
    contract for its existing Phase 1 caller (retrieve.py)."""
    model.eval()
    all_image_emb, all_text_emb, all_labels = [], [], []
    for images, texts, labels, _pids in dataloader:
        images = images.to(device)
        all_image_emb.append(model.encode_image(images).cpu().numpy())
        all_text_emb.append(model.encode_text(texts, device).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_image_emb), np.concatenate(all_text_emb), np.concatenate(all_labels)
