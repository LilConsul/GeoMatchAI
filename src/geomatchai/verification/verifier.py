from typing import Tuple

import torch
import torch.nn.functional as F


class LandmarkVerifier:
    """
    Core verification system for visual place verification.

    Implements combined similarity scoring using cosine similarity and
    normalized Euclidean distance against a reference gallery.
    """

    def __init__(self, gallery_embeddings: torch.Tensor, t_verify: float = 0.8):
        """
        Initialize the verifier with reference gallery and threshold.

        Args:
            gallery_embeddings: Reference gallery matrix of shape (N, D)
            t_verify: Verification threshold (default 0.8)
        """
        self.gallery = gallery_embeddings
        self.t_verify = t_verify

    def verify(self, query_embedding: torch.Tensor) -> Tuple[bool, float]:
        """
        Verify a query embedding against the reference gallery.

        Args:
            query_embedding: Query feature vector of shape (1, D) or (D,)

        Returns:
            Tuple of (is_verified: bool, max_score: float)
        """
        # Ensure query is 2D
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        # Cosine similarity (measures angle/direction)
        cos_sim = F.cosine_similarity(query_embedding, self.gallery, dim=1)

        # Euclidean distance (measures physical closeness/magnitude)
        euclidean_dist = torch.norm(query_embedding - self.gallery, dim=1)

        # Normalize Euclidean distance to similarity score [0, 1]
        # Higher values = more similar (closer distance)
        euclidean_sim = 1 / (1 + euclidean_dist)

        # Combined score: weighted average
        # Cosine: 60% weight, Euclidean: 40% weight
        combined_scores = 0.6 * cos_sim + 0.4 * euclidean_sim

        # Take maximum similarity score across all gallery items
        max_score = combined_scores.max().item()

        # Decision based on threshold
        is_verified = max_score > self.t_verify

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
