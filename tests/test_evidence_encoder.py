import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.classifier import NoRetrievalClassifier
from src.models.evidence_encoder import EvidenceEncoder
from src.models.fusion_classifier import RetrievalFusionClassifier


# --- label-only mode (original Arm 3, must stay reproducible) ---

def test_label_only_output_shape():
    batch, embed_dim, k, num_labels = 4, 16, 5, 14
    enc = EvidenceEncoder(num_labels, embed_dim)
    query = torch.randn(batch, embed_dim)
    evidence = torch.randn(batch, k, num_labels)
    out = enc(query, evidence)
    assert out.shape == (batch, embed_dim)


def test_label_only_is_sensitive_to_evidence():
    torch.manual_seed(0)
    batch, embed_dim, k, num_labels = 2, 16, 5, 14
    enc = EvidenceEncoder(num_labels, embed_dim)
    query = torch.randn(batch, embed_dim)
    evidence_a = torch.randn(batch, k, num_labels)
    evidence_b = torch.randn(batch, k, num_labels)
    out_a = enc(query, evidence_a)
    out_b = enc(query, evidence_b)
    assert not torch.allclose(out_a, out_b)


def test_fusion_classifier_label_only_forward_shape():
    batch, k, num_labels = 2, 5, 14
    model = RetrievalFusionClassifier(num_labels=num_labels, pretrained=False)
    images = torch.randn(batch, 3, 224, 224)
    evidence = torch.randn(batch, k, num_labels)
    logits = model(images, evidence)
    assert logits.shape == (batch, num_labels)


# --- label + image mode (Arm 3b) ---

def test_label_and_image_requires_image_evidence():
    batch, embed_dim, k, num_labels = 2, 16, 5, 14
    enc = EvidenceEncoder(num_labels, embed_dim, use_image_embeddings=True)
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)
    try:
        enc(query, evidence_labels)  # no image embeddings passed
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_label_and_image_output_shape():
    batch, embed_dim, k, num_labels = 4, 16, 5, 14
    enc = EvidenceEncoder(num_labels, embed_dim, use_image_embeddings=True)
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)
    evidence_images = torch.randn(batch, k, embed_dim)
    out = enc(query, evidence_labels, evidence_images)
    assert out.shape == (batch, embed_dim)


def test_label_and_image_is_sensitive_to_image_evidence_specifically():
    # holding labels fixed, changing only the image evidence must change the output --
    # otherwise image_proj/combine would be dead weight, same failure mode as Arm 2's retrieval
    torch.manual_seed(0)
    batch, embed_dim, k, num_labels = 2, 16, 5, 14
    enc = EvidenceEncoder(num_labels, embed_dim, use_image_embeddings=True)
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)
    images_a = torch.randn(batch, k, embed_dim)
    images_b = torch.randn(batch, k, embed_dim)
    out_a = enc(query, evidence_labels, images_a)
    out_b = enc(query, evidence_labels, images_b)
    assert not torch.allclose(out_a, out_b)


def test_image_embed_dim_can_differ_from_fusion_embed_dim():
    # neighbor embeddings don't have to be the same width as the query encoder's output
    batch, embed_dim, image_embed_dim, k, num_labels = 2, 16, 768, 5, 14
    enc = EvidenceEncoder(num_labels, embed_dim, use_image_embeddings=True, image_embed_dim=image_embed_dim)
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)
    evidence_images = torch.randn(batch, k, image_embed_dim)
    out = enc(query, evidence_labels, evidence_images)
    assert out.shape == (batch, embed_dim)


def test_fusion_classifier_with_image_embeddings_forward_shape():
    batch, k, num_labels, image_embed_dim = 2, 5, 14, 768
    model = RetrievalFusionClassifier(
        num_labels=num_labels, pretrained=False,
        use_image_embeddings=True, image_embed_dim=image_embed_dim,
    )
    images = torch.randn(batch, 3, 224, 224)
    evidence_labels = torch.randn(batch, k, num_labels)
    evidence_images = torch.randn(batch, k, image_embed_dim)
    logits = model(images, evidence_labels, evidence_images)
    assert logits.shape == (batch, num_labels)


# --- joint attention mode (Arm 3c): image and label kept as separate tokens ---

def test_joint_attention_requires_use_image_embeddings():
    try:
        EvidenceEncoder(num_labels=14, embed_dim=16, joint_attention=True)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_joint_attention_output_shape():
    batch, embed_dim, k, num_labels = 4, 16, 5, 14
    enc = EvidenceEncoder(num_labels, embed_dim, use_image_embeddings=True,
                           joint_attention=True, max_k=8)
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)
    evidence_images = torch.randn(batch, k, embed_dim)
    out = enc(query, evidence_labels, evidence_images)
    assert out.shape == (batch, embed_dim)


def test_joint_attention_is_sensitive_to_image_evidence_specifically():
    torch.manual_seed(0)
    batch, embed_dim, k, num_labels = 2, 16, 5, 14
    enc = EvidenceEncoder(num_labels, embed_dim, use_image_embeddings=True,
                           joint_attention=True, max_k=8)
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)
    images_a = torch.randn(batch, k, embed_dim)
    images_b = torch.randn(batch, k, embed_dim)
    out_a = enc(query, evidence_labels, images_a)
    out_b = enc(query, evidence_labels, images_b)
    assert not torch.allclose(out_a, out_b)


def test_joint_attention_is_sensitive_to_label_evidence_specifically():
    torch.manual_seed(0)
    batch, embed_dim, k, num_labels = 2, 16, 5, 14
    enc = EvidenceEncoder(num_labels, embed_dim, use_image_embeddings=True,
                           joint_attention=True, max_k=8)
    query = torch.randn(batch, embed_dim)
    evidence_images = torch.randn(batch, k, embed_dim)
    labels_a = torch.randn(batch, k, num_labels)
    labels_b = torch.randn(batch, k, num_labels)
    out_a = enc(query, labels_a, evidence_images)
    out_b = enc(query, labels_b, evidence_images)
    assert not torch.allclose(out_a, out_b)


def test_joint_attention_differs_from_early_fusion_given_same_weights_seed():
    # same inputs, same random seed at construction -- the two mechanisms must
    # produce different outputs, otherwise joint_attention is a no-op wrapper
    batch, embed_dim, k, num_labels = 2, 16, 5, 14
    query = torch.randn(batch, embed_dim)
    evidence_labels = torch.randn(batch, k, num_labels)
    evidence_images = torch.randn(batch, k, embed_dim)

    torch.manual_seed(1)
    early_fusion = EvidenceEncoder(num_labels, embed_dim, use_image_embeddings=True, joint_attention=False)
    torch.manual_seed(1)
    joint = EvidenceEncoder(num_labels, embed_dim, use_image_embeddings=True, joint_attention=True, max_k=8)

    out_early = early_fusion(query, evidence_labels, evidence_images)
    out_joint = joint(query, evidence_labels, evidence_images)
    assert not torch.allclose(out_early, out_joint)


def test_fusion_classifier_joint_attention_forward_shape():
    batch, k, num_labels, image_embed_dim = 2, 5, 14, 768
    model = RetrievalFusionClassifier(
        num_labels=num_labels, pretrained=False,
        use_image_embeddings=True, image_embed_dim=image_embed_dim,
        joint_attention=True, max_k=8,
    )
    images = torch.randn(batch, 3, 224, 224)
    evidence_labels = torch.randn(batch, k, num_labels)
    evidence_images = torch.randn(batch, k, image_embed_dim)
    logits = model(images, evidence_labels, evidence_images)
    assert logits.shape == (batch, num_labels)


# --- encoder-sharing fix: freeze a fusion classifier's encoder to Arm 1's weights ---

def test_load_and_freeze_encoder_copies_weights_exactly():
    num_labels = 14
    source = NoRetrievalClassifier(num_labels=num_labels, pretrained=False)
    target = RetrievalFusionClassifier(num_labels=num_labels, pretrained=False,
                                        use_image_embeddings=True, joint_attention=True, max_k=8)

    target.load_and_freeze_encoder(source.encoder.state_dict())

    for p_source, p_target in zip(source.encoder.parameters(), target.encoder.parameters()):
        assert torch.allclose(p_source, p_target)


def test_load_and_freeze_encoder_makes_encoder_untrainable():
    num_labels = 14
    source = NoRetrievalClassifier(num_labels=num_labels, pretrained=False)
    target = RetrievalFusionClassifier(num_labels=num_labels, pretrained=False,
                                        use_image_embeddings=True, joint_attention=True, max_k=8)

    target.load_and_freeze_encoder(source.encoder.state_dict())

    assert all(not p.requires_grad for p in target.encoder.parameters())
    # evidence encoder and head must remain trainable -- only the encoder is frozen
    assert any(p.requires_grad for p in target.evidence_encoder.parameters())
    assert all(p.requires_grad for p in target.head.parameters())


def test_frozen_encoder_stays_in_eval_mode_across_train_calls():
    num_labels = 14
    source = NoRetrievalClassifier(num_labels=num_labels, pretrained=False)
    target = RetrievalFusionClassifier(num_labels=num_labels, pretrained=False,
                                        use_image_embeddings=True, joint_attention=True, max_k=8)
    target.load_and_freeze_encoder(source.encoder.state_dict())

    target.train()  # this is exactly what a training loop calls every epoch
    assert not target.encoder.training
    assert target.evidence_encoder.training
    assert target.head.training


def test_frozen_encoder_forward_pass_produces_no_encoder_gradients():
    batch, k, num_labels, image_embed_dim = 2, 5, 14, 768
    source = NoRetrievalClassifier(num_labels=num_labels, pretrained=False)
    target = RetrievalFusionClassifier(num_labels=num_labels, pretrained=False,
                                        use_image_embeddings=True, image_embed_dim=image_embed_dim,
                                        joint_attention=True, max_k=8)
    target.load_and_freeze_encoder(source.encoder.state_dict())
    target.train()

    images = torch.randn(batch, 3, 224, 224)
    evidence_labels = torch.randn(batch, k, num_labels)
    evidence_images = torch.randn(batch, k, image_embed_dim)

    logits = target(images, evidence_labels, evidence_images)
    logits.sum().backward()

    assert all(p.grad is None for p in target.encoder.parameters())
    assert any(p.grad is not None for p in target.evidence_encoder.parameters())
    assert all(p.grad is not None for p in target.head.parameters())


# --- confidence gating: per-sample evidence trust cutoff ---

def test_confidence_gating_omitted_matches_all_ones():
    # backward compatibility: no evidence_confidence arg -> identical to
    # every existing Arm 3/3b/3c/3c-frozen test's behavior
    torch.manual_seed(0)
    batch, k, num_labels = 3, 5, 14
    model = RetrievalFusionClassifier(num_labels=num_labels, pretrained=False)
    images = torch.randn(batch, 3, 224, 224)
    evidence = torch.randn(batch, k, num_labels)

    out_without_arg = model(images, evidence)
    out_with_all_ones = model(images, evidence, evidence_confidence=torch.ones(batch))
    assert torch.allclose(out_without_arg, out_with_all_ones)


def test_confidence_gating_zero_gives_pure_vision_only_prediction():
    # confidence=0 must be mathematically identical to skipping evidence
    # entirely (query_embedding alone through the head) -- NOT the same as
    # zeroing the evidence input tensor. label_proj is an nn.Linear with a
    # bias, so a zeroed evidence tensor still produces a non-zero attention
    # context; only zeroing evidence_context itself (what evidence_confidence
    # does) gives a true vision-only prediction.
    torch.manual_seed(0)
    batch, k, num_labels = 2, 5, 14
    model = RetrievalFusionClassifier(num_labels=num_labels, pretrained=False)
    images = torch.randn(batch, 3, 224, 224)
    evidence = torch.randn(batch, k, num_labels)

    out_gated = model(images, evidence, evidence_confidence=torch.zeros(batch))
    out_vision_only = model.head(model.encoder(images))
    assert torch.allclose(out_gated, out_vision_only, atol=1e-6)


def test_confidence_gating_mixed_batch_is_per_sample():
    # the core requirement: some samples gated, some not, in ONE forward call
    torch.manual_seed(0)
    batch, k, num_labels = 4, 5, 14
    model = RetrievalFusionClassifier(num_labels=num_labels, pretrained=False)
    images = torch.randn(batch, 3, 224, 224)
    evidence = torch.randn(batch, k, num_labels)
    confidence = torch.tensor([1.0, 0.0, 1.0, 0.0])

    out_mixed = model(images, evidence, evidence_confidence=confidence)
    out_all_on = model(images, evidence, evidence_confidence=torch.ones(batch))
    out_all_off = model(images, evidence, evidence_confidence=torch.zeros(batch))

    # gated-off rows (1, 3) must match the all-off run; gated-on rows (0, 2) must match the all-on run
    assert torch.allclose(out_mixed[[1, 3]], out_all_off[[1, 3]], atol=1e-5)
    assert torch.allclose(out_mixed[[0, 2]], out_all_on[[0, 2]], atol=1e-5)
    # and the two groups must actually differ from each other -- otherwise gating is a no-op
    assert not torch.allclose(out_mixed[[0, 2]], out_mixed[[1, 3]])


def test_confidence_gating_composes_with_joint_attention_and_image_evidence():
    # gating must work with every existing EvidenceEncoder mode, not just label-only
    torch.manual_seed(0)
    batch, k, num_labels, image_embed_dim = 4, 5, 14, 768
    model = RetrievalFusionClassifier(
        num_labels=num_labels, pretrained=False,
        use_image_embeddings=True, image_embed_dim=image_embed_dim,
        joint_attention=True, max_k=8,
    )
    images = torch.randn(batch, 3, 224, 224)
    evidence_labels = torch.randn(batch, k, num_labels)
    evidence_images = torch.randn(batch, k, image_embed_dim)
    confidence = torch.tensor([1.0, 0.0, 1.0, 0.0])

    logits = model(images, evidence_labels, evidence_images, evidence_confidence=confidence)
    assert logits.shape == (batch, num_labels)


def test_confidence_gating_composes_with_frozen_encoder():
    # gating must still work on top of load_and_freeze_encoder (Arm 3c-frozen's setup)
    num_labels = 14
    source = NoRetrievalClassifier(num_labels=num_labels, pretrained=False)
    target = RetrievalFusionClassifier(num_labels=num_labels, pretrained=False,
                                        use_image_embeddings=True, joint_attention=True, max_k=8)
    target.load_and_freeze_encoder(source.encoder.state_dict())

    images = torch.randn(3, 3, 224, 224)
    evidence_labels = torch.randn(3, 8, num_labels)
    evidence_images = torch.randn(3, 8, 768)
    confidence = torch.tensor([1.0, 0.0, 1.0])

    logits = target(images, evidence_labels, evidence_images, evidence_confidence=confidence)
    assert logits.shape == (3, num_labels)
