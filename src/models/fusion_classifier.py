"""Arm 3 (CXR-Recall): pre-diagnosis retrieval fusion. Same image encoder
as Arm 1/2, but the classification head consumes the query embedding fused
with retrieved evidence, not the query embedding alone -- this is what
makes retrieval causally relevant to the diagnosis.

use_image_embeddings=False, joint_attention=False -> original Arm 3 (label-only).
use_image_embeddings=True,  joint_attention=False -> Arm 3b (early fusion).
use_image_embeddings=True,  joint_attention=True  -> Arm 3c (joint attention
over separate image/label tokens). See RESULTS.md for what each variant found.

All three variants above train their own encoder end-to-end, which confounds
the AUROC comparison against Arm 1 -- some of the gap could reflect the
encoder adapting differently under fusion-mediated gradients, not only the
fusion mechanism itself (see RESULTS.md, "Known deviations"). Call
load_and_freeze_encoder() with Arm 1's trained encoder state dict to remove
that confound: the encoder becomes byte-identical to Arm 1's and untrainable,
so any remaining AUROC gap is attributable purely to the fusion mechanism."""

import torch
import torch.nn as nn

from .encoders import ImageEncoder
from .evidence_encoder import EvidenceEncoder


class RetrievalFusionClassifier(nn.Module):
    def __init__(self, num_labels: int, pretrained: bool = True,
                 use_image_embeddings: bool = False, image_embed_dim: int = None,
                 joint_attention: bool = False, max_k: int = 16):
        super().__init__()
        self.encoder = ImageEncoder(pretrained=pretrained)
        self.evidence_encoder = EvidenceEncoder(
            num_labels, self.encoder.embed_dim,
            use_image_embeddings=use_image_embeddings,
            image_embed_dim=image_embed_dim,
            joint_attention=joint_attention,
            max_k=max_k,
        )
        self.head = nn.Linear(self.encoder.embed_dim, num_labels)
        self._encoder_frozen = False

    def load_and_freeze_encoder(self, encoder_state_dict: dict) -> None:
        """Loads a pretrained encoder's weights (e.g. Arm 1's best checkpoint)
        and freezes them so this classifier's own training cannot adapt them
        -- isolates the evidence-fusion mechanism as the only thing under
        test. self.encoder is kept in eval() mode from here on, including
        across .train() calls (see train() override below), since a frozen
        encoder should never see dropout/batchnorm's train-mode behavior."""
        self.encoder.load_state_dict(encoder_state_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self._encoder_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()  # override: a frozen encoder never re-enters train mode
        return self

    def forward(self, images: torch.Tensor, evidence_labels: torch.Tensor,
                evidence_image_embeddings: torch.Tensor = None,
                evidence_confidence: torch.Tensor = None) -> torch.Tensor:
        """evidence_labels: (batch, k, num_labels). evidence_image_embeddings:
        (batch, k, image_embed_dim), required iff the encoder was built with
        use_image_embeddings=True -- both precomputed via Arm 2's FAISS index
        (see retrieve.py / the notebook's Arm-3 retrieval cell).

        evidence_confidence: optional (batch,) tensor, typically hard 0.0/1.0
        from a similarity-threshold cutoff (see
        src/retrieval/confidence.py:confidence_mask) -- 0.0 means this
        query's retrieved evidence isn't trusted, so its contribution to the
        fused representation is zeroed and the prediction falls back to
        vision-only (query_embedding alone) for that sample; 1.0 means
        evidence is used normally. evidence_context is still computed for
        every sample in the batch either way -- masking happens after, since
        a vectorized batch can't skip per-sample compute -- so this only
        changes the fused representation, not the cost of getting there.
        Omitted (default None): behavior is unchanged from before this
        parameter existed, so every existing Arm 3/3b/3c/3c-frozen result
        stays reproducible without passing anything new."""
        query_embedding = self.encoder(images)
        evidence_context = self.evidence_encoder(query_embedding, evidence_labels, evidence_image_embeddings)
        if evidence_confidence is not None:
            fused = query_embedding + evidence_context * evidence_confidence.unsqueeze(-1)
        else:
            fused = query_embedding + evidence_context  # residual fusion (unchanged default path)
        return self.head(fused)
