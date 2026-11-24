"""Phase 2: text modality. Fuses the report text (already carried by
IUXrayDataset but unused through Phase 1) alongside the image, via the same
residual-sum fusion style as Arm 3's `query_embedding + evidence_context`
(see fusion_classifier.py).

Deliberately NOT three separately-trained models (vision-only, text-only,
fusion) -- one fusion model is trained, then each modality is ablated at eval
time (use_image=False / use_text=False), the same decomposed-ablation pattern
already validated for Arm 3's evidence. Two reasons: (1) it directly reuses
the ablation machinery/interpretation this project already trusts, and (2) it
sidesteps a circularity risk a separately-trained text-only model would walk
into -- IU X-ray's `mesh_tags` labels are NLM's own MeSH indexing of the
report text, so a text-only model scoring well would partly be reading its
own label source, not demonstrating clinical value. Vision-only is just Arm 1,
already trained; no need to retrain it here.

use_image=False / use_text=False replace that modality's contribution with a
zero vector *before* summation -- and, when the flag is off, the corresponding
encoder is never called at all, so ablation on a held-out set is cheap (skips
a full ClinicalBERT or ViT pass over data whose contribution is being zeroed
anyway)."""

import torch
import torch.nn as nn

from .encoders import ImageEncoder, TextEncoder


class MultimodalFusionClassifier(nn.Module):
    def __init__(self, num_labels: int, pretrained: bool = True,
                 text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.text_encoder = TextEncoder(model_name=text_model_name)
        embed_dim = self.image_encoder.embed_dim
        self.text_proj = (nn.Identity() if self.text_encoder.embed_dim == embed_dim
                           else nn.Linear(self.text_encoder.embed_dim, embed_dim))
        self.head = nn.Linear(embed_dim, num_labels)

    def forward(self, images: torch.Tensor = None, texts: list = None,
                device: torch.device = None,
                use_image: bool = True, use_text: bool = True) -> torch.Tensor:
        if not use_image and not use_text:
            raise ValueError("at least one modality must be used")
        embed_dim = self.image_encoder.embed_dim

        if use_image:
            image_embedding = self.image_encoder(images)
            batch, device = image_embedding.size(0), image_embedding.device
        if use_text:
            text_embedding = self.text_proj(self.text_encoder(texts, device))
            batch, device = text_embedding.size(0), text_embedding.device

        if not use_image:
            image_embedding = torch.zeros(batch, embed_dim, device=device)
        if not use_text:
            text_embedding = torch.zeros(batch, embed_dim, device=device)

        fused = image_embedding + text_embedding
        return self.head(fused)  # logits, (batch, num_labels)
