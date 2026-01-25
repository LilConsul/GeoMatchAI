"""
Image preprocessing module for person segmentation and removal.

Uses semantic segmentation (DeepLabV3) to detect and remove people from images,
allowing the feature extractor to focus on landmark features.
"""

import logging
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet101,
)

from ..config import config, get_effective_device

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, device: str | None = None, target_size: tuple[int, int] | None = None):
        """
        Initialize Preprocessor with person segmentation model.

        Args:
            device: Device to use ("cuda" or "cpu").
                   If None, uses config.get_device() or auto-detects.
            target_size: Target size for image resizing (H, W).
                        If None, uses config.preprocessing.TARGET_SIZE.
                        This should match the feature extractor's expected input size.
        """
        # Priority: instance parameter > global config > auto-detect
        self.device = get_effective_device(device)

        # Set target size (use provided or default from config)
        self.target_size = target_size or config.preprocessing.TARGET_SIZE

        logger.info(f"Initializing Preprocessor on device: {self.device}, target size: {self.target_size}")
        self.model = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, progress=True
        ).to(self.device)
        self.model.eval()
        logger.info("DeepLabV3 segmentation model loaded successfully")

        # Transforms for input image (resize to target size as per DeepLabV3 input)
        self.transform = T.Compose(
            [
                T.Resize(self.target_size),
                T.ToTensor(),
                T.Normalize(mean=config.preprocessing.MEAN, std=config.preprocessing.STD),
            ]
        )

        # Separate normalize transform for output consistency
        self.normalize = T.Normalize(mean=config.preprocessing.MEAN, std=config.preprocessing.STD)

    def _validate_image_dimensions(self, image: Image.Image) -> None:
        """
        Validate image dimensions for security.

        Args:
            image: PIL Image to validate.

        Raises:
            ValueError: If dimensions are invalid.
        """
        if (
            image.width > config.preprocessing.MAX_DIMENSION
            or image.height > config.preprocessing.MAX_DIMENSION
        ):
            raise ValueError(
                f"Image dimensions too large: {image.width}x{image.height} "
                f"(max {config.preprocessing.MAX_DIMENSION}x{config.preprocessing.MAX_DIMENSION})"
            )

        if (
            image.width < config.preprocessing.MIN_DIMENSION
            or image.height < config.preprocessing.MIN_DIMENSION
        ):
            raise ValueError(
                f"Image dimensions too small: {image.width}x{image.height} "
                f"(min {config.preprocessing.MIN_DIMENSION}x{config.preprocessing.MIN_DIMENSION})"
            )

    def segment_person(self, image: Image.Image) -> torch.Tensor:
        """
        Perform semantic segmentation to detect the 'person' class.

        Args:
            image: PIL Image of the user's selfie.

        Returns:
            Binary mask tensor (1 for person, 0 elsewhere) of shape (H, W).
        """
        with torch.no_grad():
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)["out"]
            # Get the predicted class for each pixel
            pred = torch.argmax(output.squeeze(), dim=0)
            # Create binary mask for person class
            mask = (pred == config.preprocessing.PERSON_CLASS_IDX).float()
        return mask

    def apply_mask(self, image: Image.Image, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the person mask using Neutral Mean Replacement method.

        Replaces person pixels with the image's per-channel mean value,
        effectively removing person features while maintaining color distribution.

        Args:
            image: Original PIL Image.
            mask: Binary mask tensor from segmentation (H, W).

        Returns:
            Cleaned image tensor of shape (3, H, W) with person removed.
        """
        # Resize and convert to tensor
        resized_image = T.Resize(self.target_size)(image)
        image_tensor = T.ToTensor()(resized_image).to(self.device)

        # Calculate per-channel mean (better color preservation than global mean)
        # Shape: (3,) for RGB channels
        channel_means = image_tensor.mean(dim=(1, 2), keepdim=True)

        # Create replacement tensor filled with per-channel means
        # Shape: (3, H, W)
        mean_filled_tensor = channel_means.expand_as(image_tensor)

        # Expand mask to 3 channels to match image tensor
        # Shape: (3, H, W)
        mask_expanded = mask.unsqueeze(0).repeat(3, 1, 1)

        # Replace person pixels (mask=1) with mean, keep background (mask=0) as original
        cleaned_tensor = torch.where(mask_expanded.bool(), mean_filled_tensor, image_tensor)

        # Apply normalization to match the transform output format
        return self.normalize(cleaned_tensor)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Full pipeline: Segment person and apply mask to remove them.

        Args:
            image: PIL Image of the user's selfie.

        Returns:
            Pre-processed image tensor (3, H, W) with person removed,
            where H, W are determined by config.preprocessing.TARGET_SIZE.

        Raises:
            ValueError: If image cannot be processed.
        """
        self._validate_image_dimensions(image)

        mask = self.segment_person(image)
        cleaned_tensor = self.apply_mask(image, mask)
        return cleaned_tensor

    def preprocess_image_from_path(self, image_path: str) -> torch.Tensor:
        """
        Convenience method: Load image from file path and preprocess.

        Args:
            image_path: Path to the user's selfie image.

        Returns:
            Pre-processed image tensor (3, H, W) with person removed,
            where H, W are determined by config.preprocessing.TARGET_SIZE.

        Raises:
            ValueError: If image cannot be loaded or processed.
            FileNotFoundError: If image file doesn't exist.
        """
        # Security: Validate file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Security: Validate file size to prevent OOM attacks
        file_size = os.path.getsize(image_path)
        max_size_bytes = config.preprocessing.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(
                f"Image too large: {file_size / (1024 * 1024):.1f}MB "
                f"(max {config.preprocessing.MAX_IMAGE_SIZE_MB}MB)"
            )

        if file_size == 0:
            raise ValueError("Image file is empty")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}") from e

        return self.preprocess_image(image)

    def transform_image(self, image: Image.Image) -> torch.Tensor:
        """
        Apply the standard transforms to a PIL image.

        Args:
            image: PIL Image to transform.

        Returns:
            Transformed tensor (3, H, W) where H, W are determined by
            config.preprocessing.TARGET_SIZE.
        """
        return self.transform(image)
