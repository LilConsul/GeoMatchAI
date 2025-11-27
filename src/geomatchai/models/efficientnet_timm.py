"""
EfficientNet feature extractors using TIMM library.

Provides better pre-trained weights than torchvision for landmark recognition.
"""

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EfficientNetFeatureExtractor(nn.Module):
    """
    EfficientNet feature extractor using timm library.

    Supports better pre-trained weights than torchvision:
    - tf_efficientnet_b4.ns_jft_in1k: NoisyStudent training (better generalization)
    - tf_efficientnet_b4.ap_in1k: AdvProp training (better robustness)
    """

    def __init__(self, model_variant="tf_efficientnet_b4.ns_jft_in1k"):
        """
        Initialize feature extractor with timm models.

        Args:
            model_variant: Model architecture. Options:
                - 'tf_efficientnet_b4.ns_jft_in1k': NoisyStudent (RECOMMENDED)
                - 'tf_efficientnet_b4.ap_in1k': AdvProp
                - 'tf_efficientnet_b4': Standard
        """
        super().__init__()

        # Create model without classification head (num_classes=0)
        self.model = timm.create_model(
            model_variant,
            pretrained=True,
            num_classes=0,  # Remove classifier, return features only
        )

        # Set to eval mode
        self.model.eval()

        # Freeze parameters for inference
        for param in self.model.parameters():
            param.requires_grad = False

        # Get feature dimension from model
        self.feature_dim = self.model.num_features  # 1792 for B4

        logger.info(f"Loaded {model_variant} with {self.feature_dim}D features")

    def forward(self, x):
        """
        Extract and normalize features.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            L2-normalized features (B, feature_dim)
        """
        features = self.model(x)
        return F.normalize(features, p=2, dim=1)


class LandmarkEfficientNet(nn.Module):
    """
    EfficientNet specifically for landmark recognition.

    Uses ensemble of models or specialized landmark-trained weights.
    """

    def __init__(self):
        super().__init__()
        # Option 1: Use multiple models and average embeddings
        self.models = nn.ModuleList(
            [
                timm.create_model("tf_efficientnet_b4.ns_jft_in1k", pretrained=True, num_classes=0),
                timm.create_model("tf_efficientnet_b4.ap_in1k", pretrained=True, num_classes=0),
            ]
        )

        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        self.feature_dim = self.models[0].num_features

    def forward(self, x):
        """Extract features using ensemble averaging."""
        embeddings = []
        for model in self.models:
            emb = model(x)
            embeddings.append(F.normalize(emb, p=2, dim=1))

        # Average normalized embeddings
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        # Re-normalize
        return F.normalize(avg_embedding, p=2, dim=1)


# Example usage:
if __name__ == "__main__":
    import torch

    # Test the model
    model = EfficientNetFeatureExtractor("tf_efficientnet_b4.ns_jft_in1k")

    # Random input (batch_size=2, RGB, 380x380)
    x = torch.randn(2, 3, 380, 380)

    with torch.no_grad():
        features = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature norm: {features.norm(dim=1)}")  # Should be ~1.0 (normalized)
