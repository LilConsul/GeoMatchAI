from typing import AsyncGenerator

import torch
from PIL import Image

from ..models.efficientnet import EfficientNetFeatureExtractor
from ..preprocessing.segmentation import Preprocessor


class GalleryBuilder:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.preprocessor = Preprocessor(device=self.device)
        self.feature_extractor = EfficientNetFeatureExtractor(model_variant="b4").to(
            self.device
        )

    async def build_gallery(
        self,
        image_generator: AsyncGenerator[Image.Image, None],
        batch_size: int = 32,
        skip_preprocessing: bool = False,
    ) -> torch.Tensor:
        """
        Process all reference images and return N×D embedding matrix.

        Args:
            image_generator: Async generator yielding reference images
            batch_size: Maximum batch size to prevent OOM (default 32)
            skip_preprocessing: Skip person removal for clean gallery images (default False)
                               Set to True if gallery photos don't have people - preserves features!

        Returns:
            Gallery embeddings tensor of shape (N, D)
        """
        all_embeddings = []
        processed_images = []

        async for image in image_generator:
            try:
                if skip_preprocessing:
                    # Apply transforms WITHOUT person removal (for clean gallery images)
                    img_tensor = self.preprocessor.transform(image).to(self.device)
                else:
                    # Use full preprocessing WITH person removal (for selfies)
                    mask = self.preprocessor.segment_person(image)
                    img_tensor = self.preprocessor.apply_mask(image, mask)

                processed_images.append(img_tensor)

                # Batch process when batch_size is reached
                if len(processed_images) == batch_size:
                    batch = torch.stack(processed_images).to(self.device)

                    # Extract features
                    with torch.no_grad():
                        embeddings = self.feature_extractor(batch)

                    all_embeddings.append(embeddings.cpu())  # Move to CPU to save GPU memory
                    processed_images = []  # Reset for next batch

                    # Clean up GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: Failed to process image: {e}")
                continue

        # Process any remaining images in the last batch
        if processed_images:
            batch = torch.stack(processed_images).to(self.device)

            # Extract features
            with torch.no_grad():
                embeddings = self.feature_extractor(batch)

            all_embeddings.append(embeddings.cpu())  # Move to CPU to save GPU memory

        if not all_embeddings:
            raise ValueError("No images could be processed successfully")

        return torch.cat(all_embeddings, dim=0).to(self.device)  # Shape: (N, D)
