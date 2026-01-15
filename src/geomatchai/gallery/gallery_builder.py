"""
Gallery builder for creating reference image embeddings.

This module builds a gallery of reference embeddings from a set of images
that can be used for landmark verification.
"""

import logging
from collections.abc import AsyncGenerator

import torch
from PIL import Image

from ..config import config, get_effective_device
from ..preprocessing.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class GalleryBuilder:
    def __init__(
        self,
        device: str | None = None,
        model_type: str = config.model.DEFAULT_MODEL_TYPE,
        model_variant: str | None = None,
    ):
        """
        Initialize GalleryBuilder with configurable feature extractor.

        Args:
            device: Device to run on ('cuda' or 'cpu').
                   If None, uses config.get_device() or auto-detects.
            model_type: Type of model to use (defaults to config.model.DEFAULT_MODEL_TYPE):
                - 'torchvision': Models from torchvision (ResNet, DenseNet, EfficientNet, etc.)
                - 'timm': Models from timm library (EfficientNets, ViT, ConvNeXt, etc.)
                - 'timm_ensemble': Ensemble of multiple timm models (slower but better)
            model_variant: Specific variant for the selected model_type
                (defaults based on model_type from config):

                For 'torchvision':
                    - 'b4': EfficientNet-B4 (default)
                    - 'resnet50', 'resnet101', 'resnet152': ResNet variants
                    - 'densenet121', 'densenet169': DenseNet variants
                    - 'mobilenet_v3_large': MobileNetV3-Large
                    - 'inception_v3': InceptionV3

                For 'timm':
                    EfficientNets:
                    - 'tf_efficientnet_b4.ns_jft_in1k': NoisyStudent (RECOMMENDED)
                    - 'tf_efficientnet_b4.ap_in1k': AdvProp
                    - 'tf_efficientnet_b4': Standard
                    - 'tf_efficientnet_b5', 'tf_efficientnet_b6'

                    Modern CNNs:
                    - 'resnest50d', 'resnest101e': ResNeSt variants
                    - 'regnety_400mf', 'regnety_8gf': RegNet variants
                    - 'convnext_base', 'convnext_large': ConvNeXt variants
                    - 'nfnet_f0': NFNet-F0

                    Vision Transformers:
                    - 'vit_base_patch16_224', 'vit_large_patch16_224': ViT variants
                    - 'deit_base_distilled_patch16_224': DeiT-Base
                    - 'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224': Swin Transformer

                    CLIP models:
                    - 'clip_vit_b32', 'clip_vit_b16', 'clip_rn50'
        """
        # Priority: instance parameter > global config > auto-detect
        self.device = get_effective_device(device)

        self.model_type = model_type
        self.model_variant = model_variant
        self.preprocessor = Preprocessor(device=self.device)

        # Load appropriate feature extractor
        if model_type == "torchvision":
            from ..models.efficientnet import EfficientNetFeatureExtractor

            variant = model_variant or config.model.DEFAULT_TORCHVISION_VARIANT
            self.feature_extractor = EfficientNetFeatureExtractor(model_variant=variant).to(
                self.device
            )
            logger.info(f"Using torchvision EfficientNet-{variant}")

        elif model_type == "timm":
            from ..models.efficientnet_timm import EfficientNetFeatureExtractor

            variant = model_variant or config.model.DEFAULT_TIMM_VARIANT
            self.feature_extractor = EfficientNetFeatureExtractor(model_variant=variant).to(
                self.device
            )
            logger.info(f"Using timm model: {variant}")

        elif model_type == "timm_ensemble":
            from ..models.efficientnet_timm import LandmarkEfficientNet

            self.feature_extractor = LandmarkEfficientNet().to(self.device)
            logger.info("Using timm ensemble (NoisyStudent + AdvProp)")

        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Choose from: 'torchvision', 'timm', 'timm_ensemble'"
            )

    async def build_gallery(
        self,
        image_generator: AsyncGenerator[Image.Image],
        batch_size: int = config.gallery.DEFAULT_BATCH_SIZE,
        skip_preprocessing: bool = False,
    ) -> torch.Tensor:
        """
        Process all reference images and return N×D embedding matrix.

        Args:
            image_generator: Async generator yielding reference images
            batch_size: Maximum batch size to prevent OOM
                       (defaults to config.gallery.DEFAULT_BATCH_SIZE)
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
                    img_tensor = self.preprocessor.preprocess_image(image)

                processed_images.append(img_tensor)

                # Batch process when batch_size is reached
                if len(processed_images) == batch_size:
                    batch_size_here = len(processed_images)
                    embeddings = self._process_batch(processed_images)
                    all_embeddings.append(embeddings)  # Move to CPU to save GPU memory
                    processed_images = []  # Reset for next batch
                    total_processed += batch_size_here
            except Exception:
                continue

        # Process any remaining images in the last batch
        if processed_images:
            batch_size_here = len(processed_images)
            embeddings = self._process_batch(processed_images)
            all_embeddings.append(embeddings)  # Move to CPU to save GPU memory
            total_processed += batch_size_here
            logger.debug(f"Processed final batch: {batch_size_here} images")

        if not all_embeddings:
            logger.error("No images could be processed successfully")
            raise ValueError("No images could be processed successfully")

        gallery_tensor = torch.cat(all_embeddings, dim=0).to(self.device)  # Shape: (N, D)
        logger.info(
            f"Gallery built successfully: {gallery_tensor.shape[0]} images, {gallery_tensor.shape[1]}D features"
        )
        return gallery_tensor

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

    def extract_embedding(
        self, image: Image.Image, skip_preprocessing: bool = False
    ) -> torch.Tensor:
        """
        Extract embedding from a single image.

        This is a convenience method for extracting embeddings from individual images,
        such as user selfies for verification.

        Args:
            image: PIL Image to extract embedding from
            skip_preprocessing: Skip person removal for clean images (default False)
                               Set to True if image doesn't have people in foreground

        Returns:
            Embedding tensor of shape (1, D) ready for verification
        """
        try:
            # Preprocess the image
            if skip_preprocessing:
                img_tensor = self.preprocessor.transform_image(image)
            else:
                img_tensor = self.preprocessor.preprocess_image(image)

            # Extract features with the model in eval mode
            self.feature_extractor.eval()
            with torch.no_grad():
                embedding = self.feature_extractor(img_tensor.unsqueeze(0).to(self.device))

            # Move to device for verification (verifier expects same device as gallery)
            return embedding.to(self.device)

        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            raise
