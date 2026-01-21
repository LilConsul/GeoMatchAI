"""
EfficientNet feature extractors using TIMM library.

Provides better pre-trained weights than torchvision for landmark recognition.
Supports a wide range of modern architectures including EfficientNets, ResNets,
ConvNeXt, Vision Transformers, and CLIP models.
"""

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EfficientNetFeatureExtractor(nn.Module):
    """
    Universal TIMM feature extractor supporting multiple architectures.

    Supports:
    - EfficientNets (B4, B5, B6 with various training strategies)
    - ResNeSt (50d, 101e)
    - RegNet (Y-040, Y-080)
    - ConvNeXt (Base, Large)
    - NFNet (dm_nfnet_f0)
    - Vision Transformers (ViT, DeiT, Swin)
    - CLIP models (ViT-B32, ViT-B16, RN50)
    """

    def __init__(self, model_variant="tf_efficientnet_b4.ns_jft_in1k"):
        """
        Initialize feature extractor with timm models.

        Args:
            model_variant: Model architecture. Examples:
                EfficientNets:
                - 'tf_efficientnet_b4.ns_jft_in1k': NoisyStudent (RECOMMENDED)
                - 'tf_efficientnet_b4.ap_in1k': AdvProp
                - 'tf_efficientnet_b4': Standard
                - 'tf_efficientnet_b5', 'tf_efficientnet_b6'

                Modern CNNs:
                - 'resnest50d', 'resnest101e'
                - 'regnety_040', 'regnety_080'
                - 'convnext_base', 'convnext_large'
                - 'dm_nfnet_f0'

                Vision Transformers:
                - 'vit_base_patch16_224', 'vit_large_patch16_224'
                - 'deit_base_distilled_patch16_224'
                - 'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224'

                CLIP:
                - 'vit_base_patch32_clip_224.openai' (or 'clip_vit_b32')
                - 'vit_base_patch16_clip_224.openai' (or 'clip_vit_b16')
                - 'resnet50.a1_in1k' (or 'clip_rn50')
        """
        super().__init__()

        # Handle CLIP model aliases
        clip_aliases = {
            "clip_vit_b32": "vit_base_patch32_clip_224.openai",
            "clip_vit_b16": "vit_base_patch16_clip_224.openai",
            "clip_rn50": "resnet50.a1_in1k",
        }

        actual_model_name = clip_aliases.get(model_variant, model_variant)
        self.model_variant = model_variant

        try:
            # Create model without classification head (num_classes=0)
            self.model = timm.create_model(
                actual_model_name,
                pretrained=True,
                num_classes=0,  # Remove classifier, return features only
            )
        except Exception as e:
            logger.error(f"Failed to load model '{actual_model_name}': {e}")
            logger.info("Listing similar available models:")
            similar = timm.list_models(f"*{model_variant.split('_')[0]}*", pretrained=True)
            for model_name in similar[:10]:
                logger.info(f"  - {model_name}")
            raise ValueError(
                f"Model '{model_variant}' not available in timm. "
                f"Check similar models above or visit https://huggingface.co/timm"
            ) from e

        # Set to eval mode
        self.model.eval()

        # Freeze parameters for inference
        for param in self.model.parameters():
            param.requires_grad = False

        # Get feature dimension from model
        self.feature_dim = self.model.num_features

        logger.info(f"Loaded timm model '{actual_model_name}' with {self.feature_dim}D features")

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
