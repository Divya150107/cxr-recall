"""Image and text encoders shared across all three experimental arms
(CXR-Recall proposal, Section 6.4)."""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, embed_dim: int = 768):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = vit_b_16(weights=weights)
        self.backbone = vit
        self.backbone.heads = nn.Identity()  # drop the ImageNet classification head
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)  # (batch, embed_dim)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size

    def forward(self, texts: list, device: torch.device, max_length: int = 256) -> torch.Tensor:
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(device)
        output = self.model(**tokens)
        return output.last_hidden_state[:, 0, :]  # [CLS] embedding, (batch, embed_dim)
