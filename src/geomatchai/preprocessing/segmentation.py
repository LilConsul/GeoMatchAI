import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights


class Prepocessor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, progress=True).to(self.device)
        self.model.eval()

        # COCO class index for "person" is 15 (0-indexed)
        self.person_class_idx = 15

        # Transforms for input image (resize to 520x520 as per DeepLabV3 input)
        self.transform = T.Compose(
            [
                T.Resize((520, 520)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Inverse transform for output (to revert normalization if needed, but we'll work with tensors)
        self.inv_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                )
            ]
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
        resized_image = T.Resize((520, 520))(image)
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
            mask_expanded.bool(),
            mean_filled_tensor,
            image_tensor
        )

        return cleaned_tensor

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Full pipeline: Load image, segment person, apply mask.

        Args:
            image_path: Path to the user's selfie image.

        Returns:
            Pre-processed image tensor (3, 520, 520) with person removed.

        Raises:
            ValueError: If image cannot be loaded or processed.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}") from e

        mask = self.segment_person(image)
        cleaned_tensor = self.apply_mask(image, mask)
        return cleaned_tensor
