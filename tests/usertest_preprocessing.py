import torchvision.transforms as T
from PIL import Image

from src.geomatchai.preprocessing.preprocessor import Preprocessor

if __name__ == "__main__":
    preprocessor = Preprocessor()
    # Example: Process a sample image
    image = Image.open("input/photo_2024-07-21_12-07-58.jpg").convert("RGB")
    cleaned = preprocessor.preprocess_image(image)
    print(f"Output shape: {cleaned.shape}")  # Should be torch.Size([3, 520, 520])
    # Save for inspection (optional)
    T.ToPILImage()(cleaned).save("output/cleaned_landmark.jpg")
