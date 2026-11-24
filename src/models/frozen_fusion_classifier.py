"""Phase 2 Arm 3: CXR-Recall evidence fusion on FROZEN Stage-1 image
embeddings (512-dim). New class, not an extension of RetrievalFusionClassifier
-- that class always owns and calls its own ImageEncoder (768-dim, hardcoded)
every forward pass, plus load_and_freeze_encoder()/a train()-mode override
built specifically to manage an *owned* encoder's frozen state. Here,
freezing already happened at Stage 1 and embeddings are precomputed/cached
once (see frozen_dual_tower_classifier.py's caching flow) -- there is no
encoder object for this class to own or toggle at all, so extending
RetrievalFusionClassifier would leave dead ViT parameters and irrelevant
freeze-management code sitting in the class for this mode, and would risk
the byte-for-byte reproducibility Phase 1's headline result (Arm 3c-frozen)
depends on. EvidenceEncoder itself needs zero changes -- embed_dim is
already a generic constructor arg, validated here at 512 instead of 768."""

import torch
import torch.nn as nn

from .evidence_encoder import EvidenceEncoder


class FrozenDualTowerEvidenceClassifier(nn.Module):
    def __init__(self, num_labels: int, embed_dim: int = 512,
                 use_image_embeddings: bool = False, image_embed_dim: int = None,
                 joint_attention: bool = False, max_k: int = 16):
        super().__init__()
        self.evidence_encoder = EvidenceEncoder(
            num_labels, embed_dim,
            use_image_embeddings=use_image_embeddings,
            image_embed_dim=image_embed_dim,
            joint_attention=joint_attention,
            max_k=max_k,
        )
        self.head = nn.Linear(embed_dim, num_labels)

    def forward(self, query_image_embedding: torch.Tensor, evidence_labels: torch.Tensor,
                evidence_image_embeddings: torch.Tensor = None,
                evidence_confidence: torch.Tensor = None) -> torch.Tensor:
        """query_image_embedding: (batch, embed_dim) -- precomputed, frozen
        Stage-1 image embedding, NOT raw images. evidence_labels/
        evidence_image_embeddings: precomputed neighbor evidence, same shapes
        RetrievalFusionClassifier expects. evidence_confidence: same optional
        per-sample gating contract as RetrievalFusionClassifier
        (src/retrieval/confidence.py-compatible), carried over for free
        since it's the same residual-fusion shape."""
        evidence_context = self.evidence_encoder(query_image_embedding, evidence_labels, evidence_image_embeddings)
        if evidence_confidence is not None:
            fused = query_image_embedding + evidence_context * evidence_confidence.unsqueeze(-1)
        else:
            fused = query_image_embedding + evidence_context
        return self.head(fused)
