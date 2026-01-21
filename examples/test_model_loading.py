"""
Quick test to verify all models can be loaded successfully.
This should be run before the comprehensive test suite.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geomatchai.gallery.gallery_builder import GalleryBuilder

# Test model configurations
MODEL_CONFIGS = [
    # ----------------- TorchVision classic CNNs -----------------
    ("torchvision", "resnet50", "ResNet50"),
    ("torchvision", "resnet101", "ResNet101"),
    ("torchvision", "resnet152", "ResNet152"),
    ("torchvision", "densenet121", "DenseNet121"),
    ("torchvision", "densenet169", "DenseNet169"),
    ("torchvision", "mobilenet_v3_large", "MobileNetV3-Large"),
    ("torchvision", "inception_v3", "InceptionV3"),
    # ----------------- TIMM EfficientNets -----------------
    ("timm", "tf_efficientnet_b4", "TIMM-Standard"),
    ("timm", "tf_efficientnet_b4.ap_in1k", "TIMM-AdvProp"),
    ("timm", "tf_efficientnet_b4.ns_jft_in1k", "TIMM-NoisyStudent"),
    ("timm", "tf_efficientnet_b5", "TIMM-EfficientNetB5"),
    ("timm", "tf_efficientnet_b6", "TIMM-EfficientNetB6"),
    # ----------------- TIMM modern CNNs -----------------
    ("timm", "resnest50d", "ResNeSt50"),
    ("timm", "resnest101e", "ResNeSt101"),
    ("timm", "regnety_040", "RegNetY-040"),
    ("timm", "regnety_080", "RegNetY-080"),
    ("timm", "convnext_base", "ConvNeXt-Base"),
    ("timm", "convnext_large", "ConvNeXt-Large"),
    ("timm", "dm_nfnet_f0", "NFNet-F0"),
    # ----------------- TIMM Vision Transformers -----------------
    ("timm", "vit_base_patch16_224", "ViT-Base"),
    ("timm", "vit_large_patch16_224", "ViT-Large"),
    ("timm", "deit_base_distilled_patch16_224", "DeiT-Base"),
    ("timm", "swin_base_patch4_window7_224", "Swin-Base"),
    ("timm", "swin_large_patch4_window7_224", "Swin-Large"),
    # ----------------- CLIP embeddings -----------------
    ("timm", "clip_vit_b32", "CLIP-ViT-B32"),
    ("timm", "clip_vit_b16", "CLIP-ViT-B16"),
    ("timm", "clip_rn50", "CLIP-RN50"),
]


def test_model_loading():
    """Test loading all model configurations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Testing {len(MODEL_CONFIGS)} model configurations...\n")

    success_count = 0
    failed_models = []

    for model_type, model_variant, model_name in MODEL_CONFIGS:
        try:
            print(f"Loading {model_name} ({model_type}/{model_variant})...", end=" ")

            # Try to create the builder
            builder = GalleryBuilder(
                device=device, model_type=model_type, model_variant=model_variant
            )

            # Test a forward pass with dummy data
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                output = builder.feature_extractor(dummy_input)

            print(f"✓ SUCCESS (feature_dim={builder.feature_extractor.feature_dim})")
            success_count += 1

            # Clean up
            del builder
            del dummy_input
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            failed_models.append((model_name, str(e)))

    print(f"\n{'=' * 80}")
    print(f"Results: {success_count}/{len(MODEL_CONFIGS)} models loaded successfully")
    print(f"{'=' * 80}")

    if failed_models:
        print(f"\nFailed models ({len(failed_models)}):")
        for model_name, error in failed_models:
            print(f"  - {model_name}: {error[:100]}")
        return False
    else:
        print("\n✓ All models loaded successfully!")
        return True


if __name__ == "__main__":
    import os

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    success = test_model_loading()
    sys.exit(0 if success else 1)
