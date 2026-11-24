import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.multimodal_classifier import MultimodalFusionClassifier

# Tiny dummy BERT (~4KB, random-init) instead of Bio_ClinicalBERT (~400MB) so
# these tests stay fast -- correctness of the fusion/ablation logic doesn't
# depend on which text encoder weights are loaded.
TINY_TEXT_MODEL = "hf-internal-testing/tiny-random-BertModel"


def _model(num_labels=14):
    return MultimodalFusionClassifier(num_labels=num_labels, pretrained=False,
                                       text_model_name=TINY_TEXT_MODEL)


def test_forward_shape():
    batch, num_labels = 3, 14
    model = _model(num_labels)
    images = torch.randn(batch, 3, 224, 224)
    texts = ["No acute cardiopulmonary abnormality."] * batch
    logits = model(images, texts)
    assert logits.shape == (batch, num_labels)


def test_output_is_sensitive_to_image():
    torch.manual_seed(0)
    model = _model()
    texts = ["Findings are unremarkable."] * 2
    images_a = torch.randn(2, 3, 224, 224)
    images_b = torch.randn(2, 3, 224, 224)
    out_a = model(images_a, texts)
    out_b = model(images_b, texts)
    assert not torch.allclose(out_a, out_b)


def test_output_is_sensitive_to_text():
    torch.manual_seed(0)
    model = _model()
    images = torch.randn(2, 3, 224, 224)
    texts_a = ["No acute cardiopulmonary abnormality."] * 2
    texts_b = ["Large right pleural effusion with associated atelectasis."] * 2
    out_a = model(images, texts_a)
    out_b = model(images, texts_b)
    assert not torch.allclose(out_a, out_b)


def test_use_image_false_skips_image_encoder():
    # ablation must not just zero the result -- it must skip the encoder call
    # entirely, so a None images tensor is safe to pass
    model = _model()
    texts = ["Stable postoperative changes."] * 2
    logits = model(images=None, texts=texts, use_image=False, use_text=True)
    assert logits.shape == (2, 14)


def test_use_text_false_skips_text_encoder():
    model = _model()
    images = torch.randn(2, 3, 224, 224)
    logits = model(images=images, texts=None, use_image=True, use_text=False)
    assert logits.shape == (2, 14)


def test_ablated_outputs_differ_from_full_fusion():
    # zeroing a modality must actually change predictions, otherwise the
    # fusion isn't using that modality at all (the failure mode this whole
    # ablation methodology exists to catch, per Arm 3's evidence-ablation)
    torch.manual_seed(0)
    model = _model()
    images = torch.randn(2, 3, 224, 224)
    texts = ["Cardiomegaly with pulmonary vascular congestion."] * 2

    full = model(images, texts, use_image=True, use_text=True)
    image_only = model(images, texts, use_image=True, use_text=False)
    text_only = model(images, texts, use_image=False, use_text=True)

    assert not torch.allclose(full, image_only)
    assert not torch.allclose(full, text_only)
    assert not torch.allclose(image_only, text_only)


def test_requires_at_least_one_modality():
    model = _model()
    try:
        model(images=None, texts=None, use_image=False, use_text=False)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_text_proj_identity_when_dims_match():
    # tiny-random-BertModel's hidden size is much smaller than ViT's 768,
    # so text_proj must be a real Linear here, not an accidental Identity
    model = _model()
    assert not isinstance(model.text_proj, torch.nn.Identity)


def test_gradients_flow_to_both_encoders():
    model = _model()
    images = torch.randn(2, 3, 224, 224, requires_grad=False)
    texts = ["Mild interstitial opacities."] * 2
    logits = model(images, texts)
    logits.sum().backward()
    assert any(p.grad is not None for p in model.image_encoder.parameters())
    assert any(p.grad is not None for p in model.text_encoder.parameters())
