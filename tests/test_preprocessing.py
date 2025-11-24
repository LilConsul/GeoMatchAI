import torchvision.transforms as T

from src.geomatchai.preprocessing.segmentation import Prepocessor

if __name__ == "__main__":
    preprocessor = Prepocessor()
    # Example: Process a sample image
    cleaned = preprocessor.preprocess_image("photo/test_photo/photo_2024-07-21_12-07-58.jpg")
    print(f"Output shape: {cleaned.shape}")  # Should be torch.Size([3, 520, 520])
    # Save for inspection (optional)
    T.ToPILImage()(cleaned).save("photo/res_photo/cleaned_landmark.jpg")
