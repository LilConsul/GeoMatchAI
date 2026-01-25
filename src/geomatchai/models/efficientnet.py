"""TorchVision-based feature extractors for landmark recognition."""

import logging

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)

# Model registry: (model_fn, weights_enum, feature_dim)
MODEL_REGISTRY = {
    "b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, 1792),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2, 2048),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, 1024),
    "densenet169": (models.densenet169, models.DenseNet169_Weights.IMAGENET1K_V1, 1664),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        960,
    ),
    "inception_v3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1, 2048),
}


class EfficientNetFeatureExtractor(nn.Module):
    """Feature extractor for TorchVision models (EfficientNet, ResNet, DenseNet, etc.)."""

    def __init__(self, model_variant="b4"):
        super().__init__()

        if model_variant not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_variant}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        model_fn, weights, feature_dim = MODEL_REGISTRY[model_variant]
        self.model_variant = model_variant
        self.feature_dim = feature_dim

        self.model = model_fn(weights=weights)
        self.input_size = self._get_optimal_input_size(weights)

        # Remove classifier based on architecture
        if model_variant.startswith("resnet"):
            self.model.fc = nn.Identity()
        elif model_variant.startswith("densenet"):
            self.model.classifier = nn.Identity()
        elif model_variant.startswith("mobilenet"):
            self.model.classifier = nn.Identity()
        elif model_variant == "inception_v3":
            self.model.fc = nn.Identity()
            self.model.aux_logits = False
        else:
            self.model.classifier = nn.Identity()

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(f"Loaded {model_variant}: {feature_dim}D features")

    def _get_optimal_input_size(self, weights) -> tuple[int, int]:
        """Determine optimal input size from weights config or architecture."""
        try:
            if hasattr(weights, "transforms"):
                transforms = weights.transforms()

                if hasattr(transforms, "crop_size"):
                    size = transforms.crop_size
                    if isinstance(size, (list, tuple)):
                        return (size[0], size[1])
                    return (size, size)

                if hasattr(transforms, "resize_size"):
                    size = transforms.resize_size
                    if isinstance(size, (list, tuple)):
                        return (max(size), max(size))
                    return (size, size)

            # Architecture defaults
            if "efficientnet" in self.model_variant:
                return (380, 380)
            elif "inception" in self.model_variant:
                return (299, 299)
            else:
                return (224, 224)
        except Exception:
            return (224, 224)

    def forward(self, x):
        """Extract and L2-normalize features."""
        return F.normalize(self.model(x), p=2, dim=1)
