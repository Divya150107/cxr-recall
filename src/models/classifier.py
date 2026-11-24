"""Arm 1 (no-retrieval baseline) from the CXR-Recall proposal, Section 6.6.
Image encoder -> linear/sigmoid multi-label head. Arms 2 and 3 (global-retrieval
and, in the ASG-Rad companion project, anatomically-constrained retrieval) reuse
this same ImageEncoder + head shape, adding an evidence-fusion stage in front
of the head — see the proposals for the full designs."""

import torch
import torch.nn as nn

from .encoders import ImageEncoder


class NoRetrievalClassifier(nn.Module):
    def __init__(self, num_labels: int, pretrained: bool = True):
        super().__init__()
        self.encoder = ImageEncoder(pretrained=pretrained)
        self.head = nn.Linear(self.encoder.embed_dim, num_labels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.head(features)  # logits, (batch, num_labels) — apply sigmoid outside for probs
