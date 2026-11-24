"""Evidence encoder + cross-attention fusion for Arm 3 (CXR-Recall). Takes
the query's own image embedding plus information about its k retrieved
neighbors (from Arm 2's FAISS index) and produces a fused representation
that the classification head consumes -- this is what makes retrieval
causally relevant to the diagnosis, unlike Arm 2's display-only retrieval.

Three evidence modes, in order of what was actually tried (see RESULTS.md):
  - label-only (Arm 3): each neighbor is represented purely by its 14-dim
    multi-hot label vector. Causally used but net *hurt* accuracy -- ~70% of
    retrieved neighbors don't share a label with the query, and label-only
    evidence gives the model no way to tell a strong visual match from a
    weak one before trusting it.
  - early fusion, label+image (Arm 3b): each neighbor's image embedding
    (already computed during retrieval) and label vector are concatenated
    and projected into ONE combined vector per neighbor *before* attention
    runs. Closed ~82% of Arm 3's gap to Arm 1. The image embedding is passed
    through its own learned `image_proj` rather than used raw, as a
    mitigation for the "shared representation" collapse risk noted in the
    KNN literature (sharing retrieval and classification representations
    directly can hurt performance).
  - joint attention, label+image (Arm 3c): image and label information for
    each neighbor are kept as SEPARATE attendable tokens (2k tokens total,
    not k pre-fused ones), tagged with a learned modality embedding (so the
    model knows which token is an image vs. a label) and a learned
    neighbor-position embedding (so the model knows which image/label pair
    belongs to the same neighbor). This lets the query attend to a specific
    neighbor's image independently of that neighbor's label, rather than
    having the two pre-blended by a fixed linear combination before
    attention ever runs -- a strictly more expressive mechanism than early
    fusion, at the cost of more attendable items (2k vs k) on a small
    dataset, which is worth watching for overfitting."""

import torch
import torch.nn as nn


class EvidenceEncoder(nn.Module):
    def __init__(self, num_labels: int, embed_dim: int, num_heads: int = 4,
                 use_image_embeddings: bool = False, image_embed_dim: int = None,
                 joint_attention: bool = False, max_k: int = 16):
        super().__init__()
        self.use_image_embeddings = use_image_embeddings
        self.joint_attention = joint_attention
        self.label_proj = nn.Linear(num_labels, embed_dim)

        if use_image_embeddings:
            image_embed_dim = image_embed_dim or embed_dim
            self.image_proj = nn.Linear(image_embed_dim, embed_dim)
            if joint_attention:
                self.modality_embed = nn.Embedding(2, embed_dim)  # 0 = image, 1 = label
                self.neighbor_pos_embed = nn.Embedding(max_k, embed_dim)
            else:
                self.combine = nn.Linear(embed_dim * 2, embed_dim)
        elif joint_attention:
            raise ValueError("joint_attention=True requires use_image_embeddings=True")

        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query_embedding: torch.Tensor, evidence_labels: torch.Tensor,
                evidence_image_embeddings: torch.Tensor = None) -> torch.Tensor:
        """query_embedding: (batch, embed_dim). evidence_labels: (batch, k, num_labels).
        evidence_image_embeddings: (batch, k, image_embed_dim), required iff
        use_image_embeddings=True -- the retrieved neighbors' own image
        embeddings, precomputed outside this module during retrieval."""
        label_embed = self.label_proj(evidence_labels)  # (batch, k, embed_dim)

        if self.use_image_embeddings:
            if evidence_image_embeddings is None:
                raise ValueError("use_image_embeddings=True requires evidence_image_embeddings")
            image_embed = self.image_proj(evidence_image_embeddings)  # (batch, k, embed_dim)

            if self.joint_attention:
                k = evidence_labels.shape[1]
                device = evidence_labels.device
                pos = self.neighbor_pos_embed(torch.arange(k, device=device)).unsqueeze(0)  # (1, k, embed_dim)
                image_type = self.modality_embed.weight[0]  # (embed_dim,)
                label_type = self.modality_embed.weight[1]  # (embed_dim,)
                image_tok = image_embed + pos + image_type
                label_tok = label_embed + pos + label_type
                evidence_embed = torch.cat([image_tok, label_tok], dim=1)  # (batch, 2k, embed_dim) -- separately attendable
            else:
                evidence_embed = self.combine(torch.cat([image_embed, label_embed], dim=-1))  # (batch, k, embed_dim)
        else:
            evidence_embed = label_embed

        query = query_embedding.unsqueeze(1)  # (batch, 1, embed_dim)
        context, _weights = self.attention(query, evidence_embed, evidence_embed)
        return context.squeeze(1)  # (batch, embed_dim)
