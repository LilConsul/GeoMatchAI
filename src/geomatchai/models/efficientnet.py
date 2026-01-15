"""
EfficientNet feature extractors using TorchVision.

Standard EfficientNet-B4 implementation from torchvision with ImageNet weights.
"""

import logging

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)

# Model registry: maps model_variant to (model_fn, weights_enum, feature_dim)
MODEL_REGISTRY = {
    # EfficientNets
    "b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, 1792),

    # ResNets
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2, 2048),

    # DenseNets
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, 1024),
    "densenet169": (models.densenet169, models.DenseNet169_Weights.IMAGENET1K_V1, 1664),

    # MobileNets
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, 960),

    # InceptionV3
    "inception_v3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1, 2048),
}


class EfficientNetFeatureExtractor(nn.Module):
    """
    Universal TorchVision feature extractor supporting multiple architectures.

    Supports:
    - EfficientNet (B4)
    - ResNet (50, 101, 152)
    - DenseNet (121, 169)
    - MobileNetV3 (Large)
    - InceptionV3
    """

    def __init__(self, model_variant="b4"):
        """
        Initialize feature extractor with specified model.

        Args:
            model_variant: Model architecture name (e.g., 'resnet50', 'densenet121', 'b4')
        """
        super().__init__()

        if model_variant not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model variant: {model_variant}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        model_fn, weights, feature_dim = MODEL_REGISTRY[model_variant]
        self.model_variant = model_variant
        self.feature_dim = feature_dim

        # Load model with pretrained weights
        self.model = model_fn(weights=weights)

        # Remove classifier based on architecture type
        if model_variant.startswith("resnet"):
            # ResNet: replace fc layer
            self.model.fc = nn.Identity()
        elif model_variant.startswith("densenet"):
            # DenseNet: replace classifier
            self.model.classifier = nn.Identity()
        elif model_variant.startswith("mobilenet"):
            # MobileNetV3: replace classifier
            self.model.classifier = nn.Identity()
        elif model_variant == "inception_v3":
            # InceptionV3: replace fc and disable aux_logits
            self.model.fc = nn.Identity()
            self.model.aux_logits = False
        else:
            # EfficientNet and others: replace classifier
            self.model.classifier = nn.Identity()

        # Set to evaluation mode for consistent inference
        self.model.eval()

        # Freeze model parameters to prevent memory leaks during inference
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(f"Loaded TorchVision {model_variant} with {feature_dim}D features")

    def forward(self, x):
        """
        Extract and L2-normalize features for metric learning.

        L2 normalization is CRITICAL for:
        - Cosine similarity to work correctly
        - Fair comparison between different images
        - Better discrimination between locations
        """
        features = self.model(x)
        # L2 normalize: each feature vector has unit length
        return F.normalize(features, p=2, dim=1)
