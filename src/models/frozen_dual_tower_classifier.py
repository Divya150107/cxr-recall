"""Phase 2 Arm 1: linear probe on FROZEN Stage-1 image+text embeddings
(confirmed decision -- Stage 2 never fine-tunes the towers, preventing the
modality-collapse failure mode diagnosed in Phase 2 attempt 1,
src/models/multimodal_classifier.py). Takes precomputed 512-dim embeddings,
not raw images/texts -- there is no encoder inside this class at all, unlike
MultimodalFusionClassifier (which owns and always runs ImageEncoder+TextEncoder
end-to-end every forward pass).

Ablation zeroes a modality's contribution at the embedding level (after the
Stage-1 projection heads, before fusion). With use_gating=False (the
default), there is no intervening biased layer between "embedding" and "the
sum that feeds the single Linear head" -- zeroing the embedding tensor IS
exactly mathematically equivalent to omitting that modality from the sum,
verified directly in tests against head(other_embedding_alone).

use_gating=True replaces the plain sum with a learned scalar gate (see
RESULTS.md "Phase 2 attempt 2", additive-fusion finding: full fusion landed
BELOW text-only alone in two separate real runs, since a fixed unweighted
sum can't down-weight a noisier modality per example). Kept deliberately a
single scalar gate, not per-dimension -- this arm is a linear probe by
design (head itself is a few thousand parameters on ~6k training examples),
and a per-dimension gate would add roughly two orders of magnitude more
parameters, a real overfitting risk out of proportion with the rest of this
arm's design. THE EXACT-EQUALITY ABLATION GUARANTEE DOES NOT HOLD under
gating: head(g * image_embeds) = g*(W @ image_embeds) + b, which only
equals head(image_embeds) = W @ image_embeds + b when g == 1 -- not
guaranteed, since head is affine (has a bias) and g is itself a learned
function of both inputs, including whatever it produces when handed a zero
vector for the ablated modality. This is the direct, unavoidable cost of
adaptive per-example weighting, not a bug -- see
tests/test_frozen_dual_tower_classifier.py for the actual relationship
gating produces instead of the old exact equality."""

import torch
import torch.nn as nn


class FrozenDualTowerClassifier(nn.Module):
    def __init__(self, num_labels: int, embed_dim: int = 512, use_gating: bool = False):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_labels)
        self.use_gating = use_gating
        if use_gating:
            self.gate = nn.Linear(embed_dim * 2, 1)

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor,
                use_image: bool = True, use_text: bool = True) -> torch.Tensor:
        if not use_image and not use_text:
            raise ValueError("at least one modality must be used")
        batch, embed_dim = image_embeds.size(0), image_embeds.size(1)
        device = image_embeds.device
        image_term = image_embeds if use_image else torch.zeros(batch, embed_dim, device=device)
        text_term = text_embeds if use_text else torch.zeros(batch, embed_dim, device=device)
        if self.use_gating:
            g = torch.sigmoid(self.gate(torch.cat([image_term, text_term], dim=-1)))  # (batch, 1)
            fused = g * image_term + (1 - g) * text_term
        else:
            fused = image_term + text_term
        return self.head(fused)  # logits, (batch, num_labels)
