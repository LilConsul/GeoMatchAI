import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet101,
)


class Preprocessor:
    # Constants for validation
    MAX_IMAGE_SIZE_MB = 50
    MAX_DIMENSION = 10000
    MIN_DIMENSION = 100
    TARGET_SIZE = (520, 520)

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, progress=True
        ).to(self.device)
        self.model.eval()

        # COCO class index for "person" is 15 (0-indexed)
        self.person_class_idx = 15

        # Transforms for input image (resize to 520x520 as per DeepLabV3 input)
        self.transform = T.Compose(
            [
                T.Resize(self.TARGET_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Separate normalize transform for output consistency
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _validate_image_dimensions(self, image: Image.Image) -> None:
        """
        Validate image dimensions for security.

        Args:
            image: PIL Image to validate.

        Raises:
            ValueError: If dimensions are invalid.
        """
        if image.width > self.MAX_DIMENSION or image.height > self.MAX_DIMENSION:
            self.logger.error(
                f"Image dimensions too large: {image.width}x{image.height} (max {self.MAX_DIMENSION}x{self.MAX_DIMENSION})"
            )
            raise ValueError(
                f"Image dimensions too large: {image.width}x{image.height} (max {self.MAX_DIMENSION}x{self.MAX_DIMENSION})"
            )

        if image.width < self.MIN_DIMENSION or image.height < self.MIN_DIMENSION:
            self.logger.error(
                f"Image dimensions too small: {image.width}x{image.height} (min {self.MIN_DIMENSION}x{self.MIN_DIMENSION})"
            )
            raise ValueError(
                f"Image dimensions too small: {image.width}x{image.height} (min {self.MIN_DIMENSION}x{self.MIN_DIMENSION})"
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
            mask = (pred == self.person_class_idx).float()
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
        resized_image = T.Resize(self.TARGET_SIZE)(image)
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
        cleaned_tensor = torch.where(
            mask_expanded.bool(), mean_filled_tensor, image_tensor
        )

        # Apply normalization to match the transform output format
        return self.normalize(cleaned_tensor)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Full pipeline: Load image, segment person, apply mask.

        Args:
            image_path: Path to the user's selfie image.

        Returns:
            Pre-processed image tensor (3, 520, 520) with person removed.

        Raises:
            ValueError: If image cannot be loaded or processed.
            FileNotFoundError: If image file doesn't exist.
        """
        # Security: Validate file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Security: Validate file size to prevent OOM attacks
        file_size = os.path.getsize(image_path)
        if file_size > self.MAX_IMAGE_SIZE_MB * 1024 * 1024:  # 50MB limit
            self.logger.error(
                f"Image too large: {file_size / (1024 * 1024):.1f}MB (max {self.MAX_IMAGE_SIZE_MB}MB)"
            )
            raise ValueError(
                f"Image too large: {file_size / (1024 * 1024):.1f}MB (max {self.MAX_IMAGE_SIZE_MB}MB)"
            )

        if file_size == 0:
            self.logger.error("Image file is empty")
            raise ValueError("Image file is empty")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Failed to load image: {e}") from e

        self._validate_image_dimensions(image)

        mask = self.segment_person(image)
        cleaned_tensor = self.apply_mask(image, mask)
        return cleaned_tensor

    def transform_image(self, image: Image.Image) -> torch.Tensor:
        """
        Apply the standard transforms to a PIL image.

        Args:
            image: PIL Image to transform.

        Returns:
            Transformed tensor (3, 520, 520).
        """
        return self.transform(image)

    def preprocess_image_from_pil(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image given as a PIL Image object.

        This method serves as an alternative to preprocess_image, allowing
        direct preprocessing of PIL Images without file I/O.

        Args:
            image: PIL Image of the user's selfie.

        Returns:
            Pre-processed image tensor (3, 520, 520) with person removed.

        Raises:
            ValueError: If image cannot be processed.
        """
        self._validate_image_dimensions(image)

        mask = self.segment_person(image)
        cleaned_tensor = self.apply_mask(image, mask)
        return cleaned_tensor
