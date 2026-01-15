"""
Landmark verification module using cosine similarity.

Compares query embeddings against a reference gallery to verify
if a user is at a specific landmark location.
"""

import logging

import torch
import torch.nn.functional as F

from ..config import config

logger = logging.getLogger(__name__)


class LandmarkVerifier:
    """
    Core verification system for visual place verification.

    Uses pure cosine similarity for optimal discrimination between matching
    and non-matching landmarks. Based on empirical testing with TIMM-NoisyStudent
    model, pure cosine similarity (w_cos=1.0, w_euc=0.0) achieves:
    - 87.5% accuracy
    - 0.332 discrimination gap (Wawel: 0.78, Unrelated: 0.45)
    - Best performance with preprocessing enabled

    Reference: examples/test_weight_optimization.py
    """

    def __init__(
        self,
        gallery_embeddings: torch.Tensor,
        t_verify: float = config.verification.DEFAULT_THRESHOLD,
    ):
        """
        Initialize the verifier with reference gallery and threshold.

        Args:
            gallery_embeddings: Reference gallery matrix of shape (N, D)
            t_verify: Verification threshold (defaults to config.verification.DEFAULT_THRESHOLD)
                     Recommended range: [config.verification.MIN_THRESHOLD,
                                        config.verification.MAX_THRESHOLD]
                     - Lower threshold (config.verification.STRICT_THRESHOLD):
                       Stricter matching, fewer false positives
                     - Higher threshold (config.verification.LENIENT_THRESHOLD):
                       More lenient, fewer false negatives
        """
        self.gallery = gallery_embeddings
        self.t_verify = t_verify

    def verify(self, query_embedding: torch.Tensor) -> tuple[bool, float]:
        """
        Verify a query embedding against the reference gallery.

        Uses pure cosine similarity for maximum discrimination between
        matching and non-matching images. Cosine similarity measures the
        angle between feature vectors, which is more robust to variations
        in lighting, scale, and minor perspective changes compared to
        Euclidean distance.

        Args:
            query_embedding: Query feature vector of shape (1, D) or (D,)

        Returns:
            Tuple of (is_verified: bool, similarity_score: float)
            - is_verified: True if max similarity > threshold
            - similarity_score: Maximum cosine similarity with gallery [0, 1]
        """
        # Ensure query is 2D
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        # Pure cosine similarity (angle-based matching)
        # Empirically proven to be optimal for landmark verification:
        # - Better discrimination gap than combined metrics
        # - More robust to scale/lighting variations
        # - Faster computation (no euclidean distance calculation)
        cos_sim = F.cosine_similarity(query_embedding, self.gallery, dim=1)

        # Take maximum similarity score across all gallery items
        max_score = cos_sim.max().item()

        # Decision based on threshold
        is_verified = max_score > self.t_verify

        logger.debug(
            f"Verification result: {'VERIFIED' if is_verified else 'REJECTED'} "
            f"(score: {max_score:.4f}, threshold: {self.t_verify})"
        )

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return is_verified, max_score

    def set_threshold(self, t_verify: float):
        """Update the verification threshold."""
        self.t_verify = t_verify

    def get_gallery_size(self) -> int:
        """Get the number of reference images in gallery."""
        return self.gallery.shape[0]
