import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.frozen_fusion_classifier import FrozenDualTowerEvidenceClassifier


def test_forward_shape_at_512dim():
    # explicitly at Phase 2's real dimension, not the tests' usual toy size --
    # validates the "EvidenceEncoder's generic embed_dim needs zero changes" claim end-to-end
    batch, embed_dim, k, num_labels = 4, 512, 5, 14
    model = FrozenDualTowerEvidenceClassifier(num_labels=num_labels, embed_dim=embed_dim)
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)
    logits = model(query, evidence_labels)
    assert logits.shape == (batch, num_labels)


def test_composability_with_image_embeddings_and_joint_attention():
    # Phase 2's neighbor embeddings are also 512-dim, unlike Phase 1's 768 --
    # exercises image_embed_dim being independently set
    batch, embed_dim, k, num_labels = 4, 512, 5, 14
    model = FrozenDualTowerEvidenceClassifier(
        num_labels=num_labels, embed_dim=embed_dim,
        use_image_embeddings=True, image_embed_dim=512,
        joint_attention=True, max_k=8,
    )
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)
    evidence_images = torch.randn(batch, k, 512)
    logits = model(query, evidence_labels, evidence_images)
    assert logits.shape == (batch, num_labels)


def test_evidence_confidence_zero_equals_pure_query_prediction():
    batch, embed_dim, k, num_labels = 2, 512, 5, 14
    model = FrozenDualTowerEvidenceClassifier(num_labels=num_labels, embed_dim=embed_dim)
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)

    out_gated = model(query, evidence_labels, evidence_confidence=torch.zeros(batch))
    out_direct = model.head(query)
    assert torch.allclose(out_gated, out_direct, atol=1e-6)


def test_evidence_confidence_omitted_matches_all_ones():
    torch.manual_seed(0)
    batch, embed_dim, k, num_labels = 3, 512, 5, 14
    model = FrozenDualTowerEvidenceClassifier(num_labels=num_labels, embed_dim=embed_dim)
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)

    out_without_arg = model(query, evidence_labels)
    out_with_all_ones = model(query, evidence_labels, evidence_confidence=torch.ones(batch))
    assert torch.allclose(out_without_arg, out_with_all_ones)


def test_class_owns_no_image_encoder():
    # architectural-contract regression test: this class should never grow
    # an owned encoder submodule -- freezing happens at Stage 1, not here
    model = FrozenDualTowerEvidenceClassifier(num_labels=14, embed_dim=512)
    child_names = {name for name, _ in model.named_children()}
    assert child_names == {"evidence_encoder", "head"}
