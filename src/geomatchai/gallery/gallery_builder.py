from typing import AsyncGenerator
import logging

import torch
from PIL import Image

from ..models.efficientnet import EfficientNetFeatureExtractor
from ..preprocessing.preprocessor import Preprocessor


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
        all_embeddings: list[torch.Tensor] = []
        processed_images: list[torch.Tensor] = []
        total_processed: int = 0

        async for image in image_generator:
            try:
                if skip_preprocessing:
                    # Apply transforms WITHOUT person removal (for clean gallery images)
                    img_tensor = self.preprocessor.transform_image(image)
                else:
                    # Use full preprocessing WITH person removal (for selfies)
                    img_tensor = self.preprocessor.preprocess_image_from_pil(image)

                processed_images.append(img_tensor)

                # Batch process when batch_size is reached
                if len(processed_images) == batch_size:
                    batch_size_here = len(processed_images)
                    embeddings = self._process_batch(processed_images)
                    all_embeddings.append(embeddings)  # Move to CPU to save GPU memory
                    processed_images = []  # Reset for next batch
                    total_processed += batch_size_here
            except Exception as e:
                continue

        # Process any remaining images in the last batch
        if processed_images:
            batch_size_here = len(processed_images)
            embeddings = self._process_batch(processed_images)
            all_embeddings.append(embeddings)  # Move to CPU to save GPU memory
            total_processed += batch_size_here

        if not all_embeddings:
            raise ValueError("No images could be processed successfully")

        return torch.cat(all_embeddings, dim=0).to(self.device)  # Shape: (N, D)

    def _process_batch(self, batch_images: list[torch.Tensor]) -> torch.Tensor:
        """
        Process a batch of images through the feature extractor.

        Args:
            batch_images: List of image tensors to process.

        Returns:
            Embeddings tensor for the batch.
        """
        batch = torch.stack(batch_images).to(self.device)
        with torch.no_grad():
            embeddings = self.feature_extractor(batch)
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embeddings.cpu()
