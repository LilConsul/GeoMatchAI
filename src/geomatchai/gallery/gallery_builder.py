import torch
from pathlib import Path
from ..models.efficientnet import EfficientNetFeatureExtractor
from ..preprocessing.segmentation import Prepocessor


class GalleryBuilder:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.preprocessor = Prepocessor(device=self.device)
        self.feature_extractor = EfficientNetFeatureExtractor(model_variant="b4").to(
            self.device
        )

    def build_gallery(self, image_paths: list[Path]) -> torch.Tensor:
        """Process all reference images and return N×D embedding matrix"""
        # Preprocess all images
        processed_images = []
        for path in image_paths:
            img_tensor = self.preprocessor.preprocess_image(str(path))
            processed_images.append(img_tensor)

        # Batch process
        batch = torch.stack(processed_images).to(self.device)

        # Extract features
        with torch.no_grad():
            embeddings = self.feature_extractor(batch)

        return embeddings  # Shape: (N, D)