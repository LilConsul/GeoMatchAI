import torchvision.transforms as T

from src.geomatchai.preprocessing.segmentation import Preprocessor

if __name__ == "__main__":
    preprocessor = Preprocessor()
    # Example: Process a sample image
    cleaned = preprocessor.preprocess_image("input/photo_2024-07-21_12-07-58.jpg")
    print(f"Output shape: {cleaned.shape}")  # Should be torch.Size([3, 520, 520])
    # Save for inspection (optional)
    T.ToPILImage()(cleaned).save("output/cleaned_landmark.jpg")
