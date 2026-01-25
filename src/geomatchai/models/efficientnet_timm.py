"""TIMM-based feature extractors for landmark recognition."""

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EfficientNetFeatureExtractor(nn.Module):
    """Feature extractor supporting EfficientNet, ViT, ConvNeXt, CLIP, and other TIMM models."""

    def __init__(self, model_variant="tf_efficientnet_b4.ns_jft_in1k"):
        super().__init__()

        # Handle CLIP aliases
        clip_aliases = {
            "clip_vit_b32": "vit_base_patch32_clip_224.openai",
            "clip_vit_b16": "vit_base_patch16_clip_224.openai",
            "clip_rn50": "resnet50.a1_in1k",
        }
        actual_model_name = clip_aliases.get(model_variant, model_variant)
        self.model_variant = model_variant

        # Create model without classification head
        self.model = timm.create_model(actual_model_name, pretrained=True, num_classes=0)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_dim = self.model.num_features
        self.input_size = self._get_input_size()

        logger.info(
            f"Loaded {actual_model_name}: {self.feature_dim}D features, {self.input_size} input"
        )

    def _get_input_size(self) -> tuple[int, int]:
        """Determine model's expected input size from config or architecture."""
        try:
            # Try pretrained_cfg first
            if hasattr(self.model, "pretrained_cfg") and self.model.pretrained_cfg:
                if "input_size" in self.model.pretrained_cfg:
                    size = self.model.pretrained_cfg["input_size"]
                    if isinstance(size, (tuple, list)) and len(size) == 3:
                        return (size[1], size[2])

            # Try default_cfg
            if hasattr(self.model, "default_cfg") and self.model.default_cfg:
                if "input_size" in self.model.default_cfg:
                    size = self.model.default_cfg["input_size"]
                    if isinstance(size, (tuple, list)) and len(size) == 3:
                        return (size[1], size[2])

            # Architecture-specific defaults
            model_lower = self.model_variant.lower()

            if any(x in model_lower for x in ["vit", "deit", "swin", "clip", "beit"]):
                return (224, 224)

            if "efficientnet" in model_lower:
                if "b7" in model_lower:
                    return (600, 600)
                elif "b6" in model_lower:
                    return (528, 528)
                elif "b5" in model_lower:
                    return (456, 456)
                elif "b4" in model_lower:
                    return (380, 380)

            if any(x in model_lower for x in ["convnext", "nfnet", "resnest", "regnet"]):
                return (224, 224)

            return (224, 224)
        except Exception as e:
            logger.warning(f"Error detecting input size for {self.model_variant}: {e}")
            return (224, 224)

    def forward(self, x):
        """Extract and L2-normalize features."""
        return F.normalize(self.model(x), p=2, dim=1)


class LandmarkEfficientNet(nn.Module):
    """Ensemble of EfficientNet models for improved landmark recognition."""

    def __init__(self):
        super().__init__()
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
        self.input_size = self._get_input_size_from_model(self.models[0])

    def _get_input_size_from_model(self, model) -> tuple[int, int]:
        """Extract input size from a timm model."""
        try:
            if hasattr(model, "pretrained_cfg") and model.pretrained_cfg:
                if "input_size" in model.pretrained_cfg:
                    size = model.pretrained_cfg["input_size"]
                    if isinstance(size, (tuple, list)) and len(size) == 3:
                        return (size[1], size[2])
            return (380, 380)  # EfficientNet-B4 default
        except Exception:
            return (380, 380)

    def forward(self, x):
        """Extract features using ensemble averaging."""
        embeddings = [F.normalize(model(x), p=2, dim=1) for model in self.models]
        return F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=1)
