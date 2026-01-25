"""Landmark verification using cosine similarity."""

import logging

import torch
import torch.nn.functional as F

from ..config import config

logger = logging.getLogger(__name__)


class LandmarkVerifier:
    """Visual place verification using cosine similarity."""

    def __init__(
        self,
        gallery_embeddings: torch.Tensor,
        t_verify: float = config.verification.DEFAULT_THRESHOLD,
    ):
        """Initialize verifier with gallery and threshold."""
        self.gallery = gallery_embeddings
        self.t_verify = t_verify

    def verify(self, query_embedding: torch.Tensor) -> tuple[bool, float]:
        """Verify query against gallery using cosine similarity.

        Returns:
            (is_verified, max_similarity_score)
        """
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        cos_sim = F.cosine_similarity(query_embedding, self.gallery, dim=1)
        max_score = cos_sim.max().item()
        is_verified = max_score > self.t_verify

        logger.debug(f"Verification: {'PASS' if is_verified else 'FAIL'} (score={max_score:.4f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return is_verified, max_score

    def set_threshold(self, t_verify: float):
        """Update verification threshold."""
        self.t_verify = t_verify

    def get_gallery_size(self) -> int:
        """Get the number of reference images in gallery."""
        return self.gallery.shape[0]
