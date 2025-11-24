import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101


class Prepocessor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = deeplabv3_resnet101(pretrained=True, progress=True).to(self.device)
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
        Apply the person mask to the image, setting masked pixels to black.

        Args:
            image: Original PIL Image.
            mask: Binary mask tensor from segmentation.

        Returns:
            Cleaned image tensor of shape (3, H, W) with person removed.
        """
        # Convert image to tensor (resize to match mask)
        image_tensor = T.ToTensor()(T.Resize((520, 520))(image)).to(self.device)
        # Expand mask to 3 channels
        mask_expanded = mask.unsqueeze(0).repeat(3, 1, 1)
        # Apply mask: set person pixels to 0 (black)
        cleaned_tensor = image_tensor * (1 - mask_expanded)
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
