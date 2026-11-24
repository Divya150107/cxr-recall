import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.contrastive import (ProjectionHead, DualTowerContrastiveModel,
                                     ImageTowerOnly, clip_contrastive_loss)

TINY_TEXT_MODEL = "hf-internal-testing/tiny-random-BertModel"


def _model():
    return DualTowerContrastiveModel(embed_dim=512, pretrained=False, text_model_name=TINY_TEXT_MODEL)


def test_projection_head_output_shape():
    head = ProjectionHead(in_dim=768, out_dim=512)
    x = torch.randn(4, 768)
    out = head(x)
    assert out.shape == (4, 512)


def test_projection_head_output_is_unit_norm():
    head = ProjectionHead(in_dim=768, out_dim=512)
    x = torch.randn(4, 768)
    out = head(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_dual_tower_forward_shapes():
    batch = 3
    model = _model()
    images = torch.randn(batch, 3, 224, 224)
    texts = ["No acute cardiopulmonary abnormality."] * batch
    image_emb, text_emb = model(images, texts, torch.device("cpu"))
    assert image_emb.shape == (batch, 512)
    assert text_emb.shape == (batch, 512)


def test_encode_image_and_encode_text_match_forward():
    torch.manual_seed(0)
    batch = 2
    model = _model()
    images = torch.randn(batch, 3, 224, 224)
    texts = ["Findings are unremarkable."] * batch
    device = torch.device("cpu")

    image_emb_fwd, text_emb_fwd = model(images, texts, device)
    image_emb_direct = model.encode_image(images)
    text_emb_direct = model.encode_text(texts, device)

    assert torch.allclose(image_emb_fwd, image_emb_direct)
    assert torch.allclose(text_emb_fwd, text_emb_direct)


def test_clip_loss_requires_batch_of_at_least_2():
    logit_scale = torch.tensor(0.0)
    image_embeds = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    text_embeds = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    try:
        clip_contrastive_loss(image_embeds, text_embeds, logit_scale)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_clip_loss_matched_pairs_score_lower_than_mismatched():
    torch.manual_seed(0)
    logit_scale = torch.tensor(2.0)
    batch, dim = 8, 16
    image_embeds = torch.nn.functional.normalize(torch.randn(batch, dim), dim=-1)

    matched_text = image_embeds.clone()  # each image's text embedding equals its own -- perfect alignment
    mismatched_text = matched_text[torch.randperm(batch)]  # shuffled -- misaligned pairs
    # guard against an accidental identity permutation making the test flaky
    while torch.equal(mismatched_text, matched_text):
        mismatched_text = matched_text[torch.randperm(batch)]

    matched_loss = clip_contrastive_loss(image_embeds, matched_text, logit_scale)
    mismatched_loss = clip_contrastive_loss(image_embeds, mismatched_text, logit_scale)
    assert matched_loss.item() < mismatched_loss.item()


def test_clip_loss_logit_scale_changes_loss_value():
    torch.manual_seed(0)
    batch, dim = 4, 16
    image_embeds = torch.nn.functional.normalize(torch.randn(batch, dim), dim=-1)
    text_embeds = torch.nn.functional.normalize(torch.randn(batch, dim), dim=-1)

    loss_a = clip_contrastive_loss(image_embeds, text_embeds, torch.tensor(0.0))
    loss_b = clip_contrastive_loss(image_embeds, text_embeds, torch.tensor(3.0))
    assert loss_a.item() != loss_b.item()


def test_image_tower_only_adapter_matches_encode_image():
    torch.manual_seed(0)
    batch = 2
    model = _model()
    adapter = ImageTowerOnly(model)
    images = torch.randn(batch, 3, 224, 224)

    assert torch.allclose(adapter(images), model.encode_image(images))
