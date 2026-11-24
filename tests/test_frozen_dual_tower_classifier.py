import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.frozen_dual_tower_classifier import FrozenDualTowerClassifier


def test_forward_shape():
    batch, embed_dim, num_labels = 4, 512, 14
    model = FrozenDualTowerClassifier(num_labels=num_labels, embed_dim=embed_dim)
    image_embeds = torch.randn(batch, embed_dim)
    text_embeds = torch.randn(batch, embed_dim)
    logits = model(image_embeds, text_embeds)
    assert logits.shape == (batch, num_labels)


def test_use_text_false_equals_head_of_image_embeds_alone():
    # direct comparison against head(other_modality_alone) -- not a
    # zeroed-input proxy -- since there's no intervening biased layer here,
    # this must hold exactly (the false-equivalence mistake already caught
    # once for evidence_confidence gating, RESULTS.md "Phase 1f")
    batch, embed_dim = 3, 512
    model = FrozenDualTowerClassifier(num_labels=14, embed_dim=embed_dim)
    image_embeds = torch.randn(batch, embed_dim)
    text_embeds = torch.randn(batch, embed_dim)

    out_gated = model(image_embeds, text_embeds, use_text=False)
    out_direct = model.head(image_embeds)
    assert torch.allclose(out_gated, out_direct, atol=1e-6)


def test_use_image_false_equals_head_of_text_embeds_alone():
    batch, embed_dim = 3, 512
    model = FrozenDualTowerClassifier(num_labels=14, embed_dim=embed_dim)
    image_embeds = torch.randn(batch, embed_dim)
    text_embeds = torch.randn(batch, embed_dim)

    out_gated = model(image_embeds, text_embeds, use_image=False)
    out_direct = model.head(text_embeds)
    assert torch.allclose(out_gated, out_direct, atol=1e-6)


def test_ablated_outputs_differ_from_full_fusion():
    torch.manual_seed(0)
    batch, embed_dim = 2, 512
    model = FrozenDualTowerClassifier(num_labels=14, embed_dim=embed_dim)
    image_embeds = torch.randn(batch, embed_dim)
    text_embeds = torch.randn(batch, embed_dim)

    full = model(image_embeds, text_embeds)
    image_only = model(image_embeds, text_embeds, use_text=False)
    text_only = model(image_embeds, text_embeds, use_image=False)

    assert not torch.allclose(full, image_only)
    assert not torch.allclose(full, text_only)
    assert not torch.allclose(image_only, text_only)


def test_requires_at_least_one_modality():
    model = FrozenDualTowerClassifier(num_labels=14, embed_dim=512)
    image_embeds = torch.randn(2, 512)
    text_embeds = torch.randn(2, 512)
    try:
        model(image_embeds, text_embeds, use_image=False, use_text=False)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- gated fusion (opt-in, use_gating=True) ---

def test_forward_shape_with_gating():
    batch, embed_dim, num_labels = 4, 512, 14
    model = FrozenDualTowerClassifier(num_labels=num_labels, embed_dim=embed_dim, use_gating=True)
    image_embeds = torch.randn(batch, embed_dim)
    text_embeds = torch.randn(batch, embed_dim)
    logits = model(image_embeds, text_embeds)
    assert logits.shape == (batch, num_labels)


def test_gated_output_sensitive_to_each_modality():
    # a gate that's a joint function of both embeddings should make the
    # output move when EITHER modality changes, holding the other fixed --
    # unlike plain summation this isn't guaranteed algebraically, so check it
    torch.manual_seed(0)
    batch, embed_dim = 3, 512
    model = FrozenDualTowerClassifier(num_labels=14, embed_dim=embed_dim, use_gating=True)
    image_embeds = torch.randn(batch, embed_dim)
    text_embeds_a = torch.randn(batch, embed_dim)
    text_embeds_b = torch.randn(batch, embed_dim)

    out_a = model(image_embeds, text_embeds_a)
    out_b = model(image_embeds, text_embeds_b)
    assert not torch.allclose(out_a, out_b)

    image_embeds_b = torch.randn(batch, embed_dim)
    out_c = model(image_embeds_b, text_embeds_a)
    assert not torch.allclose(out_a, out_c)


def test_gated_ablation_does_not_equal_pure_head():
    # the key ablation-semantics claim: with gating on, zeroing one modality
    # does NOT reduce to head(other_alone), because head is affine (has a
    # bias) and the gate itself is a learned function of both inputs
    # (including what it does when handed a zero vector). Verify the actual
    # relationship directly instead of assuming a false equivalence -- the
    # same lesson already learned once for evidence_confidence (Phase 1f).
    torch.manual_seed(0)
    batch, embed_dim = 3, 512
    model = FrozenDualTowerClassifier(num_labels=14, embed_dim=embed_dim, use_gating=True)
    image_embeds = torch.randn(batch, embed_dim)
    text_embeds = torch.randn(batch, embed_dim)

    out_gated_ablated = model(image_embeds, text_embeds, use_text=False)
    out_direct = model.head(image_embeds)
    assert not torch.allclose(out_gated_ablated, out_direct)

    # confirm the actual relationship: fused = g * image_embeds, where g is
    # the gate's output given (image_embeds, zeros)
    zeros = torch.zeros(batch, embed_dim)
    g = torch.sigmoid(model.gate(torch.cat([image_embeds, zeros], dim=-1)))
    expected = model.head(g * image_embeds)
    assert torch.allclose(out_gated_ablated, expected, atol=1e-6)


def test_use_gating_false_by_default_matches_plain_sum():
    # explicit regression guard: omitting use_gating must reproduce today's
    # exact plain-sum behavior, not just "close" -- this is what keeps the
    # two already-documented Kaggle runs' RESULTS.md numbers reproducible
    torch.manual_seed(0)
    batch, embed_dim = 3, 512
    model = FrozenDualTowerClassifier(num_labels=14, embed_dim=embed_dim)
    assert not hasattr(model, "gate")
    image_embeds = torch.randn(batch, embed_dim)
    text_embeds = torch.randn(batch, embed_dim)
    out = model(image_embeds, text_embeds)
    expected = model.head(image_embeds + text_embeds)
    assert torch.allclose(out, expected, atol=1e-6)
