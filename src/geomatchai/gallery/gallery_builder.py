from pathlib import Path
from typing import List

import torch

from ..models.efficientnet import EfficientNetFeatureExtractor
from ..preprocessing.segmentation import Preprocessor


class GalleryBuilder:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.preprocessor = Preprocessor(device=self.device)
        self.feature_extractor = EfficientNetFeatureExtractor(model_variant="b4").to(
            self.device
        )

    def build_gallery(
        self,
        image_paths: List[Path],
        batch_size: int = 32,
        skip_preprocessing: bool = False,
    ) -> torch.Tensor:
        """
        Process all reference images and return N×D embedding matrix.

        Args:
            image_paths: List of paths to reference images
            batch_size: Maximum batch size to prevent OOM (default 32)
            skip_preprocessing: Skip person removal for clean gallery images (default False)
                               Set to True if gallery photos don't have people - preserves features!

        Returns:
            Gallery embeddings tensor of shape (N, D)
        """
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            processed_images = []

            for path in batch_paths:
                try:
                    if skip_preprocessing:
                        # Load and normalize WITHOUT person removal (for clean gallery images)
                        from PIL import Image
                        import torchvision.transforms as T

                        image = Image.open(str(path)).convert("RGB")
                        image = T.Resize((520, 520))(image)  # Match query size
                        tensor = T.ToTensor()(image)
                        tensor = T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        )(tensor).to(self.device)
                        img_tensor = tensor
                    else:
                        # Use full preprocessing WITH person removal (for selfies)
                        img_tensor = self.preprocessor.preprocess_image(str(path))

                    processed_images.append(img_tensor)
                except Exception as e:
                    print(f"Warning: Failed to process {path}: {e}")
                    continue

            if not processed_images:
                continue

            # Batch process
            batch = torch.stack(processed_images).to(self.device)

            # Extract features
            with torch.no_grad():
                embeddings = self.feature_extractor(batch)

            all_embeddings.append(embeddings.cpu())  # Move to CPU to save GPU memory

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_embeddings:
            raise ValueError("No images could be processed successfully")

        return torch.cat(all_embeddings, dim=0).to(self.device)  # Shape: (N, D)
