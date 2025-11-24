from pathlib import Path
from src.geomatchai.gallery.gallery_builder import GalleryBuilder
from src.geomatchai.preprocessing.segmentation import Prepocessor
from src.geomatchai.models.efficientnet import EfficientNetFeatureExtractor
from src.geomatchai.verification.verifier import LandmarkVerifier
import torch

if __name__ == "__main__":
    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = Prepocessor(device=device)
    feature_extractor = EfficientNetFeatureExtractor().to(device)
    builder = GalleryBuilder(device=device)

    # Build small gallery (using test image as reference)
    image_paths = [Path("input/photo_2024-07-21_12-07-58.jpg")]
    gallery_embeddings = builder.build_gallery(image_paths)
    print(f"Gallery built with shape: {gallery_embeddings.shape}")

    # Initialize verifier
    verifier = LandmarkVerifier(gallery_embeddings, t_verify=0.8)

    # Process query image (same as gallery for testing - should get high score)
    query_tensor = preprocessor.preprocess_image("input/photo_2024-07-21_12-07-58.jpg")
    with torch.no_grad():
        query_embedding = feature_extractor(query_tensor.unsqueeze(0).to(device))

    # Verify
    is_verified, max_score = verifier.verify(query_embedding)
    print(f"Verification result: {is_verified}, Score: {max_score:.4f}")

    # Test with different threshold
    verifier.set_threshold(0.9)
    is_verified_high, max_score_high = verifier.verify(query_embedding)
    print(f"With higher threshold (0.9): {is_verified_high}, Score: {max_score_high:.4f}")

    print("Verification test completed!")
