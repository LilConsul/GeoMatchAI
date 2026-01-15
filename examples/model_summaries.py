"""
Model summaries for GeoMatchAI preprocessing and feature extraction.
"""

from torchinfo import summary

from geomatchai.models.efficientnet_timm import EfficientNetFeatureExtractor
from geomatchai.preprocessing.preprocessor import Preprocessor


def main():
    print("\n" + "=" * 80)
    print("GeoMatchAI Model Summaries")
    print("=" * 80 + "\n")

    # DeepLabV3 for person segmentation
    print("\n[1] DeepLabV3-ResNet101 (Person Segmentation)")
    print("-" * 80)
    preprocessor = Preprocessor(device="cpu")
    summary(
        preprocessor.model,
        input_size=(1, 3, 520, 520),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        row_settings=["var_names"],
        verbose=1,
    )

    # NoisyStudent EfficientNet-B4
    print("\n[2] EfficientNet-B4 NoisyStudent (Feature Extraction)")
    print("-" * 80)
    model = EfficientNetFeatureExtractor(model_variant="tf_efficientnet_b4.ns_jft_in1k")
    summary(
        model,
        input_size=(1, 3, 520, 520),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        row_settings=["var_names"],
        verbose=1,
    )


if __name__ == "__main__":
    main()
