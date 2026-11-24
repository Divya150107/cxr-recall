"""Phase 2 (redesign): dual-tower contrastive pretraining, following the
published ConVIRT/CLIP/GLoRIA/BioViL/CheXzero pattern -- separate image and
text towers, each projected into a shared embedding space, aligned with a
symmetric InfoNCE loss. This directly targets the two failures diagnosed in
Phase 2 attempt 1 (src/models/multimodal_classifier.py, RESULTS.md "Phase 2
attempt 1"): modality collapse (no label-prediction gradient exists here to
let one modality starve the other) and it structurally changes what "using
text" means (representation alignment, not label prediction from report
text) -- though it does NOT by itself resolve the report-text/MeSH-label
lexical-overlap circularity risk; see RESULTS.md for why that must be
re-measured separately once Stage 2 classification arms exist.

Reuses ImageEncoder/TextEncoder from encoders.py unchanged (both output
768-dim); only the two projection heads down to 512-dim shared space are new.

The actual Stage-1 training loop (mixed-precision, LR/patience tuned
specifically for contrastive pretraining rather than reusing the shared
classification-tuned LR/PATIENCE) lives in the Kaggle notebook
(notebooks/cxr_recall_phase2_kaggle.ipynb, Section 7), not here -- this
module only defines the model/loss, since no training loop exists in src/
for Stage 1 (training happens on Kaggle; local code is model definitions +
tests only, per this project's established split)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ImageEncoder, TextEncoder


class ProjectionHead(nn.Module):
    """Linear projection + L2 normalize -- one per tower, mapping each
    tower's native output into the shared embed_dim space. Normalization
    lives here (not in the caller) so the head is independently testable
    for "is the output on the unit hypersphere" without needing the full
    dual-tower model."""
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)  # (batch, out_dim), unit norm


class DualTowerContrastiveModel(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True,
                 text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.text_encoder = TextEncoder(model_name=text_model_name)
        self.image_proj = ProjectionHead(self.image_encoder.embed_dim, embed_dim)
        self.text_proj = ProjectionHead(self.text_encoder.embed_dim, embed_dim)
        # CLIP convention: store log-scale, exponentiate at use time. Clamping
        # after each optimizer step (ln(100), CLIP's published bound) is the
        # training loop's job, not this module's -- see clip_contrastive_loss.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_proj(self.image_encoder(images))  # (batch, embed_dim)

    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        return self.text_proj(self.text_encoder(texts, device))  # (batch, embed_dim)

    def forward(self, images: torch.Tensor, texts: list, device: torch.device) -> tuple:
        return self.encode_image(images), self.encode_text(texts, device)


class ImageTowerOnly(nn.Module):
    """Adapter satisfying extract_embeddings's encoder(images)->embeddings
    contract (src/retrieval/embeddings.py), so Arm 2/3's image-only embedding
    extraction can reuse that existing, tested function unchanged instead of
    duplicating its loop."""
    def __init__(self, dual_tower_model: DualTowerContrastiveModel):
        super().__init__()
        self.model = dual_tower_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images)


def clip_contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor,
                           logit_scale: torch.Tensor) -> torch.Tensor:
    """image_embeds, text_embeds: (batch, embed_dim), L2-normalized (as
    returned by encode_image/encode_text/forward). logit_scale: the model's
    raw nn.Parameter (log-scale; this function exponentiates it). Standard
    symmetric InfoNCE: cosine-similarity matrix scaled by exp(logit_scale),
    cross-entropy in both directions (image->text, text->image) against the
    diagonal (matched pairs = positives, everything else in the batch =
    negatives), averaged.

    Requires batch_size >= 2 -- with 1 example there are no negatives and the
    loss is degenerate (the single row's cross-entropy target is itself, loss
    collapses towards 0 regardless of alignment quality, silently hiding a
    real accuracy problem instead of measuring one). Raises ValueError below
    that so callers get a clear, immediate failure rather than a silently
    meaningless number -- the training loop must therefore use
    drop_last=True on the Stage-1 train DataLoader."""
    batch_size = image_embeds.shape[0]
    if batch_size < 2:
        raise ValueError(f"clip_contrastive_loss requires batch_size >= 2 for negatives, got {batch_size}")
    logits_per_image = logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()
    targets = torch.arange(batch_size, device=image_embeds.device)
    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_text, targets)
    return (loss_i2t + loss_t2i) / 2
