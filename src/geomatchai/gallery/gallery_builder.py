"""Gallery builder for creating reference image embeddings."""

import logging
from collections.abc import AsyncGenerator

import torch
from PIL import Image

from ..config import config, get_effective_device
from ..preprocessing.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class GalleryBuilder:
    def __init__(
        self, device: str | None = None, model_type: str = "timm", model_variant: str | None = None
    ):
        """Initialize GalleryBuilder with feature extractor.

        Args:
            device: Device ("cuda" or "cpu")
            model_type: "torchvision" or "timm" or "timm_ensemble"
            model_variant: Specific model variant
        """
        self.device = get_effective_device(device)
        self.model_type = model_type
        self.model_variant = model_variant

        # Load feature extractor
        if model_type == "torchvision":
            from ..models.efficientnet import EfficientNetFeatureExtractor

            variant = model_variant or config.model.DEFAULT_TORCHVISION_VARIANT
            self.feature_extractor = EfficientNetFeatureExtractor(model_variant=variant).to(
                self.device
            )

        elif model_type == "timm":
            from ..models.efficientnet_timm import EfficientNetFeatureExtractor

            variant = model_variant or config.model.DEFAULT_TIMM_VARIANT
            self.feature_extractor = EfficientNetFeatureExtractor(model_variant=variant).to(
                self.device
            )

        elif model_type == "timm_ensemble":
            from ..models.efficientnet_timm import LandmarkEfficientNet

            self.feature_extractor = LandmarkEfficientNet().to(self.device)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Create preprocessor with model's input size
        target_size = self.feature_extractor.input_size
        self.preprocessor = Preprocessor(device=self.device, target_size=target_size)

        logger.info(
            f"GalleryBuilder: {model_type}/{model_variant or 'default'}, size={target_size}"
        )

    async def build_gallery(
        self,
        image_generator: AsyncGenerator[Image.Image, None],
        batch_size: int = config.gallery.DEFAULT_BATCH_SIZE,
        skip_preprocessing: bool = False,
    ) -> torch.Tensor:
        """Process reference images and return embeddings matrix (N, D)."""
        all_embeddings = []
        processed_images = []

        async for image in image_generator:
            try:
                img_tensor = (
                    self.preprocessor.transform_image(image)
                    if skip_preprocessing
                    else self.preprocessor.preprocess_image(image)
                )
                processed_images.append(img_tensor)

                if len(processed_images) == batch_size:
                    all_embeddings.append(self._process_batch(processed_images))
                    processed_images = []
            except Exception:
                continue

        if processed_images:
            all_embeddings.append(self._process_batch(processed_images))

        if not all_embeddings:
            raise ValueError("No valid images processed")

        gallery_tensor = torch.cat(all_embeddings, dim=0).to(self.device)
        logger.info(
            f"Gallery: {gallery_tensor.shape[0]} images, {gallery_tensor.shape[1]}D features"
        )
        return gallery_tensor

    def _process_batch(self, batch_images: list[torch.Tensor]) -> torch.Tensor:
        """Process batch through feature extractor."""
        batch = torch.stack(batch_images).to(self.device)
        with torch.no_grad():
            embeddings = self.feature_extractor(batch)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embeddings.cpu()

    def extract_embedding(
        self, image: Image.Image, skip_preprocessing: bool = False
    ) -> torch.Tensor:
        """Extract embedding from single image (1, D)."""
        img_tensor = (
            self.preprocessor.transform_image(image)
            if skip_preprocessing
            else self.preprocessor.preprocess_image(image)
        )

        with torch.no_grad():
            embedding = self.feature_extractor(img_tensor.unsqueeze(0).to(self.device))

        return embedding.to(self.device)
