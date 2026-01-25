"""Image preprocessing with person segmentation and removal."""

import logging
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, deeplabv3_resnet101

from ..config import config, get_effective_device

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, device: str | None = None, target_size: tuple[int, int] | None = None):
        """Initialize preprocessor with segmentation model.

        Args:
            device: Device to use ("cuda" or "cpu")
            target_size: Target size (H, W). Defaults to config value.
        """
        self.device = get_effective_device(device)
        self.target_size = target_size or config.preprocessing.TARGET_SIZE

        self.model = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, progress=False
        ).to(self.device)
        self.model.eval()

        self.transform = T.Compose(
            [
                T.Resize(self.target_size),
                T.ToTensor(),
                T.Normalize(mean=config.preprocessing.MEAN, std=config.preprocessing.STD),
            ]
        )
        self.normalize = T.Normalize(mean=config.preprocessing.MEAN, std=config.preprocessing.STD)

        logger.info(f"Preprocessor initialized: device={self.device}, size={self.target_size}")

    def _validate_image_dimensions(self, image: Image.Image) -> None:
        """Validate image dimensions."""
        if (
            image.width > config.preprocessing.MAX_DIMENSION
            or image.height > config.preprocessing.MAX_DIMENSION
        ):
            raise ValueError(f"Image too large: {image.width}x{image.height}")

        if (
            image.width < config.preprocessing.MIN_DIMENSION
            or image.height < config.preprocessing.MIN_DIMENSION
        ):
            raise ValueError(f"Image too small: {image.width}x{image.height}")

    def segment_person(self, image: Image.Image) -> torch.Tensor:
        """Detect person class using semantic segmentation."""
        with torch.no_grad():
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)["out"]
            pred = torch.argmax(output.squeeze(), dim=0)
            mask = (pred == config.preprocessing.PERSON_CLASS_IDX).float()
        return mask

    def apply_mask(self, image: Image.Image, mask: torch.Tensor) -> torch.Tensor:
        """Apply person mask using neutral mean replacement."""
        resized_image = T.Resize(self.target_size)(image)
        image_tensor = T.ToTensor()(resized_image).to(self.device)

        channel_means = image_tensor.mean(dim=(1, 2), keepdim=True)
        mean_filled = channel_means.expand_as(image_tensor)
        mask_expanded = mask.unsqueeze(0).repeat(3, 1, 1)
        cleaned = torch.where(mask_expanded.bool(), mean_filled, image_tensor)

        return self.normalize(cleaned)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Segment and remove person from image."""
        self._validate_image_dimensions(image)
        mask = self.segment_person(image)
        return self.apply_mask(image, mask)

    def preprocess_image_from_path(self, image_path: str) -> torch.Tensor:
        """Load image from path and preprocess."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        file_size = os.path.getsize(image_path)
        max_size = config.preprocessing.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if file_size > max_size or file_size == 0:
            raise ValueError(f"Invalid file size: {file_size / (1024 * 1024):.1f}MB")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}") from e

        return self.preprocess_image(image)

    def transform_image(self, image: Image.Image) -> torch.Tensor:
        """Apply transforms without preprocessing."""
        return self.transform(image)
